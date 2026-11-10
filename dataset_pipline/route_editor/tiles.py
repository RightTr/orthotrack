"""Crop a selected region from a Long reference DOP and create a manual tile."""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from dataset_pipline.loader import SceneLoader

ROOT=Path(__file__).resolve().parents[2]


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)


def scene_names(ref_path):
    ref_path=Path(ref_path)
    if (ref_path/'meta.json').is_file():return [ref_path.name]
    return sorted(p.name for p in ref_path.iterdir() if (p/'meta.json').is_file())


@lru_cache(maxsize=6)
def source_info(scene,ref_path):
    ref_path=Path(ref_path)
    ref=ref_path if (ref_path/'meta.json').is_file() else ref_path/scene
    if not (ref/'meta.json').is_file():raise ValueError(f'Unknown scene: {scene}')
    loader=SceneLoader(ref,load_mesh=False,load_dsm=False)
    meta=loader.meta
    year=max(int(key.removeprefix('dop_')) for key in meta['dops'])
    dop,image=loader.dop(year)
    image_path=ref/dop['file']
    camera=meta['reference_camera']
    return {'image':image,'image_path':image_path,'reference':ref,'meta':meta,
            'year':year,'bounds':dop['bounds'],'utm_offset':meta['utm_offset'],
            'agl':camera['agl_quantiles_m'],'tilt':camera['median_tilt_from_nadir_deg'],
            'fov':camera['median_fov_vertical_deg']}


def crop_candidate(scene,rect,ref_path):
    info=source_info(scene,Path(ref_path));image=info['image'];h,w=image.shape[:2]
    r=np.asarray(rect,float)
    if r.shape!=(4,) or not np.isfinite(r).all() or np.any(r<0) or np.any(r>1):
        raise ValueError('Crop rectangle must be normalized to [0,1]')
    x0,y0,x1,y1=map(int,np.rint(r*np.array([w,h,w,h])).astype(int))
    cw,ch=x1-x0,y1-y0
    if cw<1 or ch<1:raise ValueError('Crop must have positive width and height')
    xmin,ymin,xmax,ymax=map(float,info['bounds'])
    bounds=[xmin+x0/w*(xmax-xmin),ymax-y1/h*(ymax-ymin),
            xmin+x1/w*(xmax-xmin),ymax-y0/h*(ymax-ymin)]
    ok,encoded=cv2.imencode('.jpg',image[y0:y1,x0:x1])
    if not ok:raise IOError(f'Cannot encode DOP crop: {scene}')
    return {'scene':scene,'x0':x0,'y0':y0,'width':int(cw),'height':int(ch),
            'bounds_local':bounds,'jpeg':encoded.tobytes(),
            'agl_min_m':float(info['agl'][0]),'agl_m':float(info['agl'][1]),
            'agl_max_m':float(info['agl'][2]),'tilt_deg':float(info['tilt']),
            'fov':float(info['fov']),'year':int(info['year']),'utm_offset':info['utm_offset'],
            'source_dop':str(info['image_path'])}


def accept_candidate(candidate,root,ref_path):
    scene=candidate['scene'];root=Path(root);scene_dir=root if root.name==scene else root/scene;index=1
    while (scene_dir/f'tile{index:02d}').exists():index+=1
    tile_id=f'tile{index:02d}';name=f'{scene}_{tile_id}'
    info=source_info(scene,Path(ref_path));meta=info['meta'];year=candidate['year']
    tile_dir=scene_dir/tile_id;map_ref=tile_dir/'reference'
    (map_ref/'dop').mkdir(parents=True,exist_ok=True)
    (tile_dir/'dop.jpg').write_bytes(candidate['jpeg'])
    (map_ref/'dop'/f'{year}.jpg').symlink_to((tile_dir/'dop.jpg').resolve())
    for filename in ('mesh.npz','dsm.npz'):
        (map_ref/filename).symlink_to((info['reference']/filename).resolve())
    (map_ref/'render_dop.jpg').symlink_to((info['image_path']).resolve())
    bounds=candidate['bounds_local']
    gsd=(bounds[2]-bounds[0])/candidate['width']
    map_meta=dict(meta)
    map_meta['dops']={f'dop_{year}':dict(meta['dops'][f'dop_{year}'],
                                        file=f'dop/{year}.jpg',bounds=bounds,gsd=gsd)}
    map_meta['render_dop']={'file':'render_dop.jpg','bounds':info['bounds']}
    save_json(map_ref/'meta.json',map_meta)
    cfg=yaml.safe_load((ROOT/'dataset_pipline/config/config.yaml').read_text())
    cfg['sequence'].update(name=name,frames=1000,fov_vertical_deg=candidate['fov'])
    cfg['map'].update(reference_scene=str(map_ref),tile_dir=str(tile_dir),dop_year=year,
                      scene_name=name,crop_width_px=candidate['width'],
                      crop_height_px=candidate['height'])
    cfg['trajectory'].update(manual_route_file=str(tile_dir/f'{tile_id}.json'),
                             height_agl_m=candidate['agl_m'],reference_agl_m=candidate['agl_m'],
                             reference_agl_min_m=candidate['agl_min_m'],
                             reference_agl_max_m=candidate['agl_max_m'],
                             tilt_from_nadir_deg=float(np.clip(candidate['tilt_deg'],20.,30.)),
                             minimum_tilt_from_nadir_deg=20.,maximum_tilt_from_nadir_deg=30.,
                             camera_center_weight=1.,min_view_overlap_fraction=.1,max_view_crop_scale=5.,
                             turn_slowdown_distance_m=25.,max_turn_rate_deg_s=35.)
    (tile_dir/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    save_json(tile_dir/'crop.json',{'scene':scene,'tile':tile_id,'source_dop':candidate['source_dop'],
        'source_crop_xywh_px':[candidate['x0'],candidate['y0'],candidate['width'],candidate['height']],
        'width_px':candidate['width'],'height_px':candidate['height'],'bounds_local':bounds,
        'utm_offset':candidate['utm_offset'],'gsd_m_per_pixel':gsd,
        'reference_agl_m':[candidate['agl_min_m'],candidate['agl_m'],candidate['agl_max_m']]})
    state_path=root/'selected_tiles.json'
    state=json.loads(state_path.read_text()) if state_path.is_file() else {'tiles':{}}
    state['tiles'][name]={'scene':scene,'tile_dir':str(tile_dir),'status':'ready_to_draw',
                          'config':str(tile_dir/'config.yaml')}
    save_json(state_path,state)
    return name,tile_dir/'config.yaml'
