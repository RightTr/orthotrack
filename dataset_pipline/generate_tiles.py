"""Sample fixed DOP tiles per Long scene, plan trajectories across them, then render."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from collision import dsm_heights

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/Data/RIght/MovingDrone_Long')
OUT=Path('/mnt/Data/RIght/MovingDrone_Tiles')


def save_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)


def csv_rows(path):
    import csv
    with path.open() as f:
        return list(csv.DictReader(f))


def reference_heights(source, ref):
    rows=csv_rows(source/'baseline/poses.csv')
    xyz=np.asarray([[float(row[k]) for k in ('x','y','z')] for row in rows])
    with np.load(ref/'dsm.npz') as data:
        ground=dsm_heights(data['height'],data['bounds'],xyz[:,:2])
    agl=xyz[:,2]-ground
    agl=agl[np.isfinite(agl)&(ground>-9000)&(agl>0)]
    if len(agl)<30:
        raise ValueError(f'Not enough valid Long baseline AGL samples: {source}')
    return np.quantile(agl,[.05,.5,.95])


def sample_rect(rng,w,h,tw,th,used):
    for _ in range(500):
        x=int(rng.integers(0,w-tw+1)); y=int(rng.integers(0,h-th+1))
        if all(max(0,min(x+tw,ux+tw)-max(x,ux))*max(0,min(y+th,uy+th)-max(y,uy))<.1*tw*th
               for ux,uy in used):
            return x,y
    raise ValueError('Could not sample three distinct DOP regions')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',help='Long scene name; omit to process all scenes')
    p.add_argument('--limit',type=int,help='Process only the first N scenes')
    p.add_argument('--frames',type=int,default=3000)
    p.add_argument('--tiles-per-scene',type=int,default=3)
    p.add_argument('--dop-size-m',nargs=2,type=float,metavar=('WIDTH','HEIGHT'),
                   help='Fixed DOP size in metres; default is one third of each Long cropped DOP side')
    a=p.parse_args()
    cfg0=yaml.safe_load((ROOT/'dataset_pipline/config.yaml').read_text())
    sources=[DATA/a.source] if a.source else sorted(x for x in DATA.iterdir() if (x/'reference/meta.json').is_file())
    if a.source and not (sources[0]/'reference/meta.json').is_file():
        raise FileNotFoundError(sources[0]/'reference/meta.json')
    if a.limit:
        sources=sources[:a.limit]
    OUT.mkdir(parents=True,exist_ok=True)
    state_path=OUT/'generation_tiles.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get('layout')!='three-random-tiles-per-scene':
        state={'layout':'three-random-tiles-per-scene','area_fraction':1/9,'jobs':{}}

    jobs=[(source,index) for source in sources for index in range(a.tiles_per_scene)]
    for job_index,(source,tile_index) in enumerate(tqdm(jobs,desc='Sequences'),1):
        crop_path=next((source/'baseline/dop').glob('*_cropped.jpg'))
        crop_meta=json.loads(crop_path.with_suffix('.json').read_text())
        image=cv2.imread(str(crop_path))
        if image is None:
            raise ValueError(f'Cannot read {crop_path}')
        h,w=image.shape[:2]
        xmin,ymin,xmax,ymax=map(float,crop_meta['bounds'])
        gsd=(xmax-xmin)/w
        tw,th=(round(w/3),round(h/3)) if not a.dop_size_m else tuple(round(v/gsd) for v in a.dop_size_m)
        if tw>w or th>h:
            raise ValueError(f'{source.name}: requested DOP size exceeds source crop')

        name=f'{source.name}_tile{tile_index+1:02d}'
        old=state['jobs'].get(name,{})
        seed=int(old.get('seed',np.random.SeedSequence().generate_state(1)[0]))
        rng=np.random.default_rng(seed)
        used=[]
        for other_i in range(tile_index):
            other=state['jobs'].get(f'{source.name}_tile{other_i+1:02d}')
            if other:
                used.append((int(other['x0']),int(other['y0'])))
        if 'x0' in old:
            x0,y0=int(old['x0']),int(old['y0'])
        else:
            x0,y0=sample_rect(rng,w,h,tw,th,used)
        x1,y1=x0+tw,y0+th
        bounds=[xmin+x0/w*(xmax-xmin),ymax-y1/h*(ymax-ymin),
                xmin+x1/w*(xmax-xmin),ymax-y0/h*(ymax-ymin)]
        ref=source/'reference'
        meta=json.loads((ref/'meta.json').read_text())
        year=int(crop_meta['year'].removeprefix('dop_'))
        map_ref=OUT/'map_cache'/name/'reference'
        (map_ref/'dop').mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(map_ref/'dop'/f'{year}.jpg'),image[y0:y1,x0:x1])
        for filename in ('mesh.npz','dsm.npz','poses.csv'):
            target=map_ref/filename
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to((ref/filename).resolve())
        map_meta=dict(meta)
        map_meta['dops']={f'dop_{year}':dict(meta['dops'][f'dop_{year}'],
                            file=f'dop/{year}.jpg',bounds=bounds,gsd=gsd)}
        save_json(map_ref/'meta.json',map_meta)

        cfg=json.loads(json.dumps(cfg0))
        cfg['output_root']=str(OUT/source.name)
        cfg['sequence'].update(name=name,frames=a.frames)
        cfg['map'].update(reference_scene=str(map_ref),dop_year=year,
                          preserve_dop_bounds=True,fill_dop_trajectory=True,
                          trajectory_coverage_ratio=.9,margin_m=8,
                          fixed_dop_size_m=[bounds[2]-bounds[0],bounds[3]-bounds[1]])
        low_agl,ref_agl,high_agl=reference_heights(source,ref)
        height_amplitude=float((high_agl-low_agl)/2.4)
        ref_fov=float(np.median([float(r['fov_vertical']) for r in csv_rows(ref/'poses.csv')]))
        cfg['sequence']['fov_vertical_deg']=ref_fov
        traj_seed=int.from_bytes(hashlib.sha256(f'{name}:{seed}:trajectory'.encode()).digest()[:4],'little')
        cfg['trajectory'].update(height_offset_ratio=0.,height_variation_ratio=float(height_amplitude/ref_agl),
                                 min_agl_m=float(low_agl),reference_agl_m=float(ref_agl),
                                 height_agl_m=float(ref_agl),height_variation_m=float(height_amplitude),
                                 look_at_map_center=True,fit_fov_to_dop=True,min_dop_view_overlap=.98,
                                 seed=traj_seed,passes=2,max_speed_mps=12.,
                                 min_distance_m=float(min(400.,.8*((bounds[2]-bounds[0])+(bounds[3]-bounds[1])))),
                                 min_scale_ratio=1.35,max_auto_passes=30,random_center_ratio=0.)

        scene=OUT/source.name/name/'baseline'
        used_cfg=scene/'config_used.json'
        matches=False
        if used_cfg.is_file():
            used=json.loads(used_cfg.read_text())
            matches=(used['sequence']['frames']==a.frames
                     and used['trajectory']['min_agl_m']==cfg['trajectory']['min_agl_m']
                     and used['trajectory']['height_agl_m']==float(ref_agl)
                     and used['trajectory'].get('look_at_map_center') is True
                     and used['trajectory'].get('fit_fov_to_dop') is True
                     and used['trajectory']['seed']==traj_seed)
        state['jobs'][name]={'status':'running','seed':seed,'trajectory_seed':traj_seed,
                             'x0':x0,'y0':y0,'dop_size_m':cfg['map']['fixed_dop_size_m'],
                             'source':source.name,'scene':str(scene)}
        save_json(state_path,state)
        config_path=OUT/'configs'/f'{name}.yaml'
        config_path.parent.mkdir(parents=True,exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        print(f'[{job_index}/{len(jobs)}] {name}: fixed DOP {tw}x{th}px, {a.frames} frames, Long AGL p05/50/95={low_agl:.0f}/{ref_agl:.0f}/{high_agl:.0f}m',flush=True)
        if matches and (scene/'video.mp4').is_file() and (scene/'poses.csv').is_file():
            state['jobs'][name]['status']='done'
            save_json(state_path,state)
            continue
        try:
            subprocess.run([sys.executable,'dataset_pipline/pipeline.py','generate','--config',str(config_path),
                            '--variant','baseline','--keep-variant-name'],cwd=ROOT,check=True)
            subprocess.run([sys.executable,'dataset_pipline/pipeline.py','validate','--config',str(config_path),
                            '--variant','baseline','--keep-variant-name'],cwd=ROOT,check=True)
            state['jobs'][name]['status']='done'
        except subprocess.CalledProcessError as e:
            state['jobs'][name]['status']='failed'
            state['jobs'][name]['error']=str(e)
            save_json(state_path,state)
            raise
        save_json(state_path,state)
    print(f'Saved three random tiles per scene under {OUT}')


if __name__=='__main__':
    main()
