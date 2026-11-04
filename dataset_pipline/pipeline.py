"""MovingDrone long-sequence generator. Run from the OrthoTrack repository root."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
from dataset_pipline.trajectory import make_poses,export_ges
from dataset_pipline.planner import make_route,challenge_report
from dataset_pipline.collision import dsm_heights,validate as validate_collision
from utils.pose import PoseLoader


def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def check_video(path,n,cfg):
    cap=cv2.VideoCapture(str(path)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for frame in np.linspace(0,max(0,count-1),min(9,count),dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
        if not ok or np.std(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))<1.0:
            cap.release();raise ValueError(f'Video frame {frame} is blank or cannot decode: {path}')
    cap.release()
    seq=cfg['sequence']
    if count!=n or abs(fps-seq['fps'])>.05 or (width,height)!=(seq['width'],seq['height']):
        raise ValueError(f'Video mismatch {path}: {count} frames, {fps} fps, {width}x{height}; expected {n}, {seq["fps"]}, {seq["width"]}x{seq["height"]}')


def frustum_hits(local,rot,floor,cfg,dsm,bounds):
    vf=np.deg2rad(cfg['sequence']['fov_vertical_deg'])
    fy=1/np.tan(vf/2); fx=fy/(cfg['sequence']['width']/cfg['sequence']['height'])
    rays=np.array([[x/fx,y/fy,1] for x in (-1,0,1) for y in (-1,0,1)],float)
    directions=np.einsum('nij,kj->nki',rot,rays)
    invalid=np.argwhere(directions[:,:,2]>=-.05)
    if len(invalid): raise ValueError(f'Camera frustum reaches horizon at frame {int(invalid[0,0])}')
    # Intersect each ray with the actual DSM using fixed point updates from a
    # median-ground initialization. Clamp lookup coordinates only for iteration;
    # the resulting projected point remains unclamped for the DOP bounds test.
    distance=(floor-local[:,2])[:,None]/directions[:,:,2]
    h,w=dsm.shape; x0,y0,x1,y1=bounds
    for _ in range(5):
        hits=local[:,None,:]+distance[:,:,None]*directions
        col=np.clip((hits[:,:,0]-x0)/(x1-x0)*(w-1),0,w-1.001)
        row=np.clip((y1-hits[:,:,1])/(y1-y0)*(h-1),0,h-1.001)
        c=col.astype(int);r=row.astype(int);dx=col-c;dy=row-r
        surface=((1-dx)*(1-dy)*dsm[r,c]+dx*(1-dy)*dsm[r,c+1]
                 +(1-dx)*dy*dsm[r+1,c]+dx*dy*dsm[r+1,c+1])
        surface=np.where(surface>-9000,surface,floor)
        distance=(surface[:,]-local[:,2,None])/directions[:,:,2]
    hits=local[:,None,:]+distance[:,:,None]*directions
    return hits[:,:,:2]


def mesh_visibility(local,rot,cfg,mesh_path):
    import open3d as o3d
    data=np.load(mesh_path); scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(data['vertices'].astype(np.float32)),o3d.core.Tensor(data['faces'].astype(np.uint32)))
    vf=np.deg2rad(cfg['sequence']['fov_vertical_deg']); fy=1/np.tan(vf/2)
    fx=fy/(cfg['sequence']['width']/cfg['sequence']['height'])
    pixels=np.array([[x,y,1.] for x in np.linspace(-1,1,12) for y in np.linspace(-1,1,8)])
    pixels[:,0]/=fx;pixels[:,1]/=fy
    frames=np.linspace(0,len(local)-1,min(17,len(local)),dtype=int);fractions=[]
    for i in frames:
        directions=pixels@rot[i].T;directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        rays=np.column_stack((np.broadcast_to(local[i],directions.shape),directions)).astype(np.float32)
        hits=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        fractions.append(float(np.isfinite(hits).mean()))
    threshold=cfg['map'].get('min_mesh_view_fraction',.08)
    if min(fractions)<threshold:
        i=int(frames[int(np.argmin(fractions))])
        raise ValueError(f'Mesh has insufficient visible surface at frame {i}: {min(fractions):.1%}')
    return {'frames':frames.tolist(),'visible_fraction':fractions,'minimum_fraction':min(fractions)}


def prepare_forward_sweep(cfg, scene, frames):
    ref=Path(cfg['map']['reference_scene']); meta=json.loads((ref/'meta.json').read_text())
    if cfg['map']['vertical_offset_m'] is None:
        raise ValueError('Set map.vertical_offset_m after datum calibration')
    cfg=json.loads(json.dumps(cfg));cfg['sequence']['frames']=frames
    offset=np.asarray(meta['utm_offset'],float)
    with np.load(ref/'dsm.npz') as data:
        dsm=data['height'];dsm_bounds=np.asarray(data['bounds'],float)
    dop_info=meta['dops'][f'dop_{cfg["map"]["dop_year"]}']
    image=cv2.imread(str(ref/dop_info['file']))
    if image is None: raise FileNotFoundError(ref/dop_info['file'])
    H,W=image.shape[:2]; db=np.asarray(dop_info['bounds'],float)
    cw=int(cfg['map']['crop_width_px']);ch=int(cfg['map']['crop_height_px'])
    if not 0<cw<=W or not 0<ch<=H: raise ValueError(f'Crop {cw}x{ch} exceeds reference DOP {W}x{H}')
    crop_path=scene.parent/'crop.json';seed=int(cfg['trajectory']['seed'])
    if crop_path.exists():
        crop_info=json.loads(crop_path.read_text())
        if crop_info['seed']!=seed or crop_info['crop_xywh'][2:]!=[cw,ch]:
            raise ValueError(f'Existing crop differs from requested seed or size: {crop_path}')
        x0,y0=crop_info['crop_xywh'][:2]
    else:
        rng=np.random.default_rng(seed)
        x0=int(rng.integers(0,W-cw+1)) if cfg['map']['random_crop'] else (W-cw)//2
        y0=int(rng.integers(0,H-ch+1)) if cfg['map']['random_crop'] else (H-ch)//2
    if not 0<=x0<=W-cw or not 0<=y0<=H-ch: raise ValueError('Saved crop exceeds reference DOP')
    crop=image[y0:y0+ch,x0:x0+cw]
    crop_bounds=np.array([db[0]+x0/W*(db[2]-db[0]),db[3]-(y0+ch)/H*(db[3]-db[1]),
                          db[0]+(x0+cw)/W*(db[2]-db[0]),db[3]-y0/H*(db[3]-db[1])])
    gsd=(crop_bounds[2]-crop_bounds[0])/cw
    if crop_path.exists() and not np.allclose(crop_info['crop_bounds_local'],crop_bounds,atol=1e-6):
        raise ValueError('Saved crop bounds differ from reference DOP')
    crop_info={'seed':seed,'crop_xywh':[x0,y0,cw,ch],
               'crop_bounds_local':crop_bounds.tolist(),'gsd_m_per_pixel':float(gsd)}
    valid=np.isfinite(dsm)&(dsm>-9000)
    if not valid.any(): raise ValueError('DSM has no valid elevations')
    ground=np.where(valid,dsm,0.)
    cell=max((dsm_bounds[2]-dsm_bounds[0])/dsm.shape[1],(dsm_bounds[3]-dsm_bounds[1])/dsm.shape[0])
    sigma=max(1.,20./cell)
    smooth=gaussian_filter(ground,sigma)/np.maximum(gaussian_filter(valid.astype(float),sigma),1e-6)
    world_bounds=crop_bounds+np.tile(offset[:2],2)
    terrain=float(np.nanmedian(dsm[valid]))+offset[2]+cfg['map']['vertical_offset_m']
    positions=make_route(cfg,offset[:2]+(crop_bounds[:2]+crop_bounds[2:])/2,terrain,world_bounds)
    local=positions-offset
    terrain_path=dsm_heights(smooth,dsm_bounds,local[:,:2])
    roof=dsm_heights(dsm,dsm_bounds,local[:,:2])
    mismatch=terrain_path-roof
    mismatch-=.5*(mismatch.min()+mismatch.max())
    base=float(cfg['trajectory']['height_agl_m']);amp=float(cfg['trajectory']['height_variation_m'])
    usable_amp=amp-np.max(np.abs(mismatch))
    if usable_amp < 0: raise ValueError('DSM terrain variation cannot fit requested AGL range without abrupt altitude changes')
    phase_height=positions[:,2]-(terrain+base)
    positions[:,2]=roof+offset[2]+cfg['map']['vertical_offset_m']+base+mismatch+phase_height*(usable_amp/max(amp,1e-9))
    local=positions-offset
    roof=dsm_heights(dsm,dsm_bounds,local[:,:2]);agl=local[:,2]-roof
    if agl.min()<max(base-amp,cfg['trajectory']['safe_distance_m'])-1e-3 or agl.max()>base+amp+1e-3:
        raise ValueError(f'Actual DSM AGL {agl.min():.1f}..{agl.max():.1f} m exceeds requested {base-amp:.1f}..{base+amp:.1f} m; choose a flatter crop or adjust height')
    speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)*cfg['sequence']['fps']
    if speed.max()>cfg['trajectory']['max_speed_mps']*1.01:
        raise ValueError(f'Forward sweep exceeds max speed: {speed.max():.2f} m/s')
    rot=make_poses(positions,cfg)
    floor=float(np.nanmedian(dsm[valid]))
    hits=frustum_hits(local,rot,floor,cfg,dsm,dsm_bounds)
    hit_bounds=np.r_[hits.reshape(-1,2).min(0),hits.reshape(-1,2).max(0)]
    margin=cfg['map']['render_margin_ratio']*(crop_bounds[2:]-crop_bounds[:2])+cfg['map']['render_margin_m']
    render_bounds=np.r_[np.minimum(crop_bounds[:2]-margin,hit_bounds[:2]-cfg['map']['render_margin_m']),
                        np.maximum(crop_bounds[2:]+margin,hit_bounds[2:]+cfg['map']['render_margin_m'])]
    if np.any(render_bounds[:2]<dsm_bounds[:2]) or np.any(render_bounds[2:]>dsm_bounds[2:]):
        raise ValueError(f'DSM does not cover required render bounds {render_bounds.tolist()}')
    with np.load(ref/'mesh.npz') as mesh:
        mesh_xy=mesh['vertices'][:,:2];mesh_bounds=np.r_[mesh_xy.min(0),mesh_xy.max(0)]
    if np.any(render_bounds[:2]<mesh_bounds[:2]) or np.any(render_bounds[2:]>mesh_bounds[2:]):
        raise ValueError(f'Reference Mesh does not cover required render bounds {render_bounds.tolist()}')
    if np.any(render_bounds[:2]<db[:2]) or np.any(render_bounds[2:]>db[2:]):
        raise ValueError('Reference DOP texture does not cover required render bounds')
    visibility=mesh_visibility(local,rot,cfg,ref/'mesh.npz')
    collision=validate_collision(local,dsm,dsm_bounds,ref/'mesh.npz',cfg['trajectory']['safe_distance_m'])
    scene.mkdir(parents=True,exist_ok=True);atomic_json(crop_path,crop_info)
    cfg['sequence']['name']=scene.parent.name+'_'+scene.name
    doc=export_ges(scene/'trajectory.json',positions,rot,cfg)
    loader=PoseLoader(str(scene/'trajectory.json'),utm_zone=cfg['map']['utm_zone'])
    if np.max(np.linalg.norm(loader.get_all_positions_utm()-positions,axis=1))>.02:
        raise ValueError('PoseLoader position round-trip failed')
    for i in np.linspace(0,frames-1,min(frames,50),dtype=int):
        if Rotation.from_matrix(loader.poses[i].rotation_matrix @ rot[i].T).magnitude()>1e-5:
            raise ValueError(f'PoseLoader orientation round-trip failed at frame {i}')
    dop_dir=scene/'dop';dop_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(dop_dir/f'{cfg["map"]["dop_year"]}.jpg'),crop)
    shutil.copy2(ref/'dsm.npz',scene/'dsm.npz')
    vf=np.deg2rad(cfg['sequence']['fov_vertical_deg'])
    fy=cfg['sequence']['height']/(2*np.tan(vf/2))
    atomic_json(scene/'intrinsics.json',{'width':cfg['sequence']['width'],'height':cfg['sequence']['height'],
                                        'fx':fy,'fy':fy,'cx':cfg['sequence']['width']/2,'cy':cfg['sequence']['height']/2})
    atomic_json(scene/'meta.json',{'name':cfg['sequence']['name'],'width':cfg['sequence']['width'],
        'height':cfg['sequence']['height'],'num_frames':frames,'frame_rate':cfg['sequence']['fps'],
        'utm_zone':cfg['map']['utm_zone'],'utm_offset':offset.tolist(),'dsm':meta['dsm'],
        'dops':{f'dop_{cfg["map"]["dop_year"]}':{'year':cfg['map']['dop_year'],
        'file':f'dop/{cfg["map"]["dop_year"]}.jpg','bounds':crop_bounds.tolist(),'gsd':float(gsd)}},
        'render_bounds_local':render_bounds.tolist(),'vertical_offset_m':cfg['map']['vertical_offset_m']})
    atomic_json(scene/'config_used.json',cfg)
    atomic_json(scene/'mesh_visibility.json',visibility);atomic_json(scene/'collision_report.json',collision)
    report=challenge_report(cfg,local,rot,agl,1.)
    report['render_bounds_local']=render_bounds.tolist();report['crop_bounds_local']=crop_bounds.tolist()
    atomic_json(scene/'plan_report.json',report)
    preview=scene/'preview';preview.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,7))
    ax.imshow(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),extent=crop_bounds[[0,2,1,3]],origin='upper')
    ax.plot(local[:,0],local[:,1],linewidth=.5);fig.savefig(preview/'trajectory_on_dop.png',dpi=150);plt.close(fig)
    with (scene/'poses.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['frame_id','x','y','z','qw','qx','qy','qz','latitude','longitude','altitude','fov_vertical'])
        for i,p in enumerate(loader.poses):
            c=doc['cameraFrames'][i]['coordinate']
            writer.writerow([i,p.x-offset[0],p.y-offset[1],p.z-offset[2],p.qw,p.qx,p.qy,p.qz,
                             c['latitude'],c['longitude'],c['altitude']-offset[2],doc['cameraFrames'][i]['fovVertical']])
    return doc


def prepare(cfg, scene, frames):
    return prepare_forward_sweep(cfg, scene, frames)


def render(cfg,scene,doc):
    return render_cpu(cfg,scene,doc)


def render_cpu(cfg,scene,doc):
    """Raycast the reference mesh and sample its original DOP texture without EGL."""
    import open3d as o3d
    ref=Path(cfg['map']['reference_scene'])
    meta=json.loads((ref/'meta.json').read_text())
    offset=np.asarray(meta['utm_offset'],np.float32)
    with np.load(ref/'mesh.npz') as data:
        vertices=data['vertices'].astype(np.float32)
        faces=data['faces'].astype(np.uint32)
    ray_scene=o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces))
    dop=meta['dops'][f'dop_{cfg["map"]["dop_year"]}']
    image=cv2.imread(str(ref/dop['file']))
    if image is None: raise FileNotFoundError(ref/dop['file'])
    bounds=np.asarray(dop['bounds'],float)
    dh,dw=image.shape[:2];h=cfg['sequence']['height'];w=cfg['sequence']['width']
    fps=cfg['sequence']['fps'];n=len(doc['cameraFrames'])
    loader=PoseLoader(str(scene/'trajectory.json'),utm_zone=cfg['map']['utm_zone'])
    vf=np.deg2rad(cfg['sequence']['fov_vertical_deg']);f=h/(2*np.tan(vf/2))
    gy,gx=np.mgrid[:h,:w];xx=((gx-w/2)/f).astype(np.float32);yy=((gy-h/2)/f).astype(np.float32)
    rays=np.empty((h,w,6),np.float32)
    depth_dir=scene/'depth';depth_dir.mkdir(exist_ok=True)
    video=scene/'video.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened(): raise RuntimeError('Cannot open CPU video writer')
    samples=[];sample_ids=set(np.linspace(0,n-1,min(9,n),dtype=int))
    try:
        for i in range(n):
            pose=loader.get_pose(i);R=pose.rotation_matrix.astype(np.float32)
            center=(pose.position-offset).astype(np.float32)
            rays[:,:,0:3]=center
            rays[:,:,3]=xx*R[0,0]+yy*R[0,1]+R[0,2]
            rays[:,:,4]=xx*R[1,0]+yy*R[1,1]+R[1,2]
            rays[:,:,5]=xx*R[2,0]+yy*R[2,1]+R[2,2]
            rays[:,:,3:6]/=np.maximum(np.linalg.norm(rays[:,:,3:6],axis=2,keepdims=True),1e-9)
            hit=ray_scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
            valid=np.isfinite(hit);distance=np.where(valid,hit,0.)
            xy=center[:2]+rays[:,:,3:5]*distance[:,:,None]
            col=np.clip(((xy[:,:,0]-bounds[0])/(bounds[2]-bounds[0])*dw).astype(np.int32),0,dw-1)
            row=np.clip(((bounds[3]-xy[:,:,1])/(bounds[3]-bounds[1])*dh).astype(np.int32),0,dh-1)
            frame=image[row,col].copy();frame[~valid]=0
            writer.write(frame)
            if i in sample_ids: samples.append(cv2.resize(frame,(320,180)))
            if cfg['render'].get('save_depth',False):
                maximum=float(distance.max()) if valid.any() else 1.
                encoded=np.zeros((h,w),np.uint16)
                encoded[valid]=np.clip(np.rint(distance[valid]/maximum*65535),1,65535).astype(np.uint16)
                dest=depth_dir/f'depth_{i:04d}.png'
                if not cv2.imwrite(str(dest),encoded): raise IOError(f'Failed to save {dest}')
                atomic_json(dest.with_suffix('.json'),{'depth_min_m':0.,'depth_max_m':maximum})
            if (i+1)%200==0 or i+1==n: print(f'CPU render: {i+1}/{n}',flush=True)
    finally:
        writer.release()
    check_video(video,n,cfg)
    if cfg['render'].get('save_depth',False):
        atomic_json(depth_dir/'metadata.json',{'frames':n,'width':w,'height':h,'units':'metres',
            'representation':'ray_distance','dtype':'uint16','format':'png+per-frame-json','mesh':'reference/mesh.npz'})
    if samples:
        rows=[np.hstack(samples[j:j+3]) for j in range(0,len(samples),3)]
        cv2.imwrite(str(scene/'preview'/'camera_view_samples.jpg'),np.vstack(rows))


def generate_forward_all(args,cfg):
    root=Path(cfg['output_root'])
    scenes=sorted(p.name for p in root.iterdir() if (p/'reference'/'meta.json').is_file())
    if args.scene: scenes=[s for s in scenes if s==args.scene]
    if not scenes: raise ValueError('No reference scenes found')
    jobs=[]
    for name in scenes:
        for sample in range(1,int(cfg['sequence'].get('samples',1))+1):
            command=[sys.executable,str(Path(__file__).resolve()),'generate','--config',str(args.config.resolve()),
                     '--scene',name,'--sample',str(sample)]
            if args.seed is not None: command.extend(['--seed',str(args.seed)])
            result=subprocess.run(command)
            jobs.append({'scene':name,'sample':sample,'variant':'baseline','status':'done' if result.returncode==0 else 'failed'})
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/'forward_sweep_generation.json'
    atomic_json(manifest,{'status':'completed' if all(j['status']=='done' for j in jobs) else 'completed_with_failures','jobs':jobs})
    print(f'Forward sweep generation: {manifest}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['plan','generate','generate-all','validate'])
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--scene');parser.add_argument('--frames',type=int)
    parser.add_argument('--fov-scale',type=float);parser.add_argument('--tilt-deg',type=float)
    parser.add_argument('--seed',type=int);parser.add_argument('--sample',type=int,default=1)
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if args.command=='generate-all':
        generate_forward_all(args,cfg);return
    if args.scene:
        cfg['map']['reference_scene']=str(Path(cfg['output_root'])/args.scene/'reference')
    reference=Path(cfg['map']['reference_scene'])
    with (reference/'poses.csv').open() as f:
        fovs=np.asarray([float(row['fov_vertical']) for row in csv.DictReader(f)])
    fovs=fovs[np.isfinite(fovs)&(fovs>0)]
    if not len(fovs): raise ValueError(f'Reference has no valid camera FOV: {reference}')
    fov_scale=args.fov_scale if args.fov_scale is not None else 1.
    if not 0<fov_scale<=1: parser.error('--fov-scale must be in (0, 1]')
    cfg['sequence']['fov_vertical_deg']=float(np.median(fovs))*fov_scale
    if args.tilt_deg is not None: cfg['trajectory']['tilt_from_nadir_deg']=args.tilt_deg
    source_name=cfg['map'].get('scene_name',reference.parent.name)
    if args.sample<1: parser.error('--sample must be positive')
    seed=args.seed if args.seed is not None else cfg['trajectory']['seed']
    cfg['trajectory']['seed']=int.from_bytes(hashlib.sha256(f'{source_name}:{seed}:{args.sample}'.encode()).digest()[:4],'little')
    frames=args.frames or cfg['sequence']['frames']
    cfg['sequence']['frames']=frames;cfg['sequence']['name']=f'{source_name}_baseline'
    scene=Path(cfg['output_root'])/f'{source_name}_crop{args.sample:04d}'/'baseline'
    if args.command=='validate':
        check_video(scene/'video.mp4',frames,cfg)
        with (scene/'poses.csv').open() as f: rows=list(csv.DictReader(f))
        if len(rows)!=frames: raise ValueError('Pose row count mismatch')
        positions=np.asarray([[float(row[k]) for k in ('x','y','z')] for row in rows])
        if not np.isfinite(positions).all(): raise ValueError('Non-finite pose position')
        speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)*cfg['sequence']['fps']
        if len(speed) and speed.max()>cfg['trajectory']['max_speed_mps']*1.01:
            raise ValueError(f'Speed discontinuity: {speed.max():.2f} m/s')
        meta=json.loads((scene/'meta.json').read_text())
        if len(meta['utm_offset'])!=3 or len(meta['dops'])!=1: raise ValueError('Invalid map metadata')
        plan=json.loads((scene/'plan_report.json').read_text())
        low=cfg['trajectory']['height_agl_m']-cfg['trajectory']['height_variation_m']
        high=cfg['trajectory']['height_agl_m']+cfg['trajectory']['height_variation_m']
        if plan['frames']!=frames or plan['agl_min_m']<low-.1 or plan['agl_max_m']>high+.1:
            raise ValueError('Plan report violates frame or AGL constraints')
        depth=scene/'depth';pngs=sorted(depth.glob('depth_*.png')) if depth.exists() else []
        if pngs and len(pngs)!=frames: raise ValueError('Depth frame count mismatch')
        print(f'Validated {frames} frames: {scene}');return
    doc=prepare(cfg,scene,frames)
    if args.command=='plan':
        print(f'Planned {frames} frames without rendering: {scene}');return
    render(cfg,scene,doc)
    print(f'Complete: {scene}')


if __name__=='__main__': main()
