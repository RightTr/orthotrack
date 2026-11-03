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
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation

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


def fit_map_scale(route_xy,hit_xy,center,allowed,margin):
    path=route_xy-center
    footprint=hit_xy-route_xy[:,None,:]
    lo=footprint.min(axis=(0,1))-margin
    hi=footprint.max(axis=(0,1))+margin
    span=path.max(0)-path.min(0)
    scale=min(1.,float(np.min((allowed[2:]-allowed[:2]-(hi-lo))/np.maximum(span,1e-6))))
    if scale<=0: raise ValueError('Camera footprint alone is larger than the available DOP')
    center_lo=allowed[:2]-scale*path.min(0)-lo
    center_hi=allowed[2:]-scale*path.max(0)-hi
    if np.any(center_lo>center_hi+1e-5): raise ValueError('No feasible route center inside DOP bounds')
    center_hi=np.maximum(center_hi,center_lo)
    return scale,np.clip(center,center_lo,center_hi)


def lift_route(route,dsm,bounds,offset,cfg):
    roof=dsm_heights(dsm,bounds,route[:,:2]-offset[:2])+offset[2]
    lift=max(0.,float(np.max(roof+cfg['trajectory']['safe_distance_m']+.25-route[:,2])),
             float(cfg['trajectory']['min_agl_m']-np.min(route[:,2]-roof)))
    if lift>cfg['trajectory']['max_auto_lift_m']:
        raise ValueError(f'DSM requires {lift:.2f} m lift, above max_auto_lift_m')
    route[:,2]+=lift
    return roof,lift


def frustum_bbox(local,rot,floor,cfg,dsm,bounds):
    points=frustum_hits(local,rot,floor,cfg,dsm,bounds).reshape(-1,2)
    return np.r_[points.min(0)-cfg['map']['margin_m'],points.max(0)+cfg['map']['margin_m']]


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


def reference_parameters(ref,dsm,bounds):
    with (ref/'poses.csv').open() as f:
        rows=list(csv.DictReader(f))
    xyz=np.array([[float(row[k]) for k in ('x','y','z')] for row in rows])
    inside=(np.isfinite(xyz).all(axis=1)&(xyz[:,0]>=bounds[0])&(xyz[:,0]<bounds[2])
            &(xyz[:,1]>=bounds[1])&(xyz[:,1]<bounds[3]))
    surface=dsm_heights(dsm,bounds,xyz[inside,:2])
    agl=xyz[inside,2]-surface
    valid=np.isfinite(agl)&(surface>-9000)&(agl>0)
    if valid.sum()<30: raise ValueError(f'Not enough valid reference heights in {ref}')
    fov=np.array([float(row['fov_vertical']) for row in rows])
    fov=fov[np.isfinite(fov)&(fov>0)]
    return float(np.median(agl[valid])),float(np.median(fov))


def prepare(cfg, scene, frames):
    ref=Path(cfg['map']['reference_scene']); meta=json.loads((ref/'meta.json').read_text())
    if cfg['map']['vertical_offset_m'] is None:
        raise ValueError('Vertical datum unverified: set map.vertical_offset_m explicitly after DSM/mesh datum calibration')
    offset=np.asarray(meta['utm_offset'],float)
    with np.load(ref/'dsm.npz') as data: dsm=data['height']; bounds=data['bounds']
    cfg=json.loads(json.dumps(cfg))
    cfg['sequence']['frames']=max(cfg['sequence']['frames'],frames)
    dop_info=meta['dops'][f'dop_{cfg["map"]["dop_year"]}']
    db=np.asarray(dop_info['bounds']);allowed=np.array([max(bounds[0],db[0]),max(bounds[1],db[1]),min(bounds[2],db[2]),min(bounds[3],db[3])])
    safe_allowed=allowed+np.array([2.,2.,-2.,-2.])
    # Use the typical ground elevation; the 30 m map margin covers low terrain outliers.
    floor=float(np.nanmedian(dsm[dsm>-9000]))
    # Plan inside the actual satellite-map footprint, not the usually larger DSM extent.
    mid=np.array([(allowed[0]+allowed[2])/2,(allowed[1]+allowed[3])/2])
    placement_path=scene.parent/'trajectory_placement.json'
    seed_text=f'{scene.parent.name}:{cfg["trajectory"]["seed"]}:center'.encode()
    placement_seed=int.from_bytes(hashlib.sha256(seed_text).digest()[:8],'little')
    placement=json.loads(placement_path.read_text()) if placement_path.is_file() else {}
    if placement.get('seed')==placement_seed:
        center_delta=np.asarray(placement['offset_xy_m'],float)
    else:
        ratio=cfg['trajectory'].get('random_center_ratio',0.)
        seed=placement_seed
        rng=np.random.default_rng(seed)
        center_delta=rng.uniform(-ratio,ratio,2)*(allowed[2:]-allowed[:2])
        placement={'seed':seed,'randomized':bool(ratio),'offset_xy_m':center_delta.tolist()}
        atomic_json(placement_path,placement)
    center=offset[:2]+mid+center_delta
    terrain=dsm_heights(dsm,bounds,(center-offset[:2])[None])[0]
    base_passes=cfg['trajectory']['passes']
    cfg['trajectory']['passes_override']=base_passes
    base=make_route(cfg,center,terrain+offset[2]+cfg['map']['vertical_offset_m'],allowed)
    lift_route(base,dsm,bounds,offset,cfg)
    base_rot=make_poses(base,cfg)
    base_hits=frustum_hits(base-offset,base_rot,floor,cfg,dsm,bounds)
    scale,center_fit=fit_map_scale(base[:,:2]-offset[:2],base_hits,center-offset[:2],safe_allowed,cfg['map']['margin_m'])
    estimate=base.copy()
    estimate[:,:2]=offset[:2]+center_fit+scale*(base[:,:2]-center)
    estimate_roof=dsm_heights(dsm,bounds,estimate[:,:2]-offset[:2])+offset[2]
    estimate[:,2]+=max(0.,float(np.max(estimate_roof+cfg['trajectory']['safe_distance_m']+.25-estimate[:,2])),
                        float(cfg['trajectory']['min_agl_m']-np.min(estimate[:,2]-estimate_roof)))
    estimate_distance=float(np.linalg.norm(np.diff(estimate,axis=0),axis=1).sum())
    required=base_passes if estimate_distance>=cfg['trajectory']['min_distance_m'] else int(np.ceil(base_passes*cfg['trajectory']['min_distance_m']/estimate_distance*1.1))
    maximum=cfg['trajectory'].get('max_auto_passes',10)
    if required>maximum: raise ValueError(f'DOP-limited route needs {required} passes, maximum is {maximum}')
    cfg['trajectory']['passes_override']=required
    base=make_route(cfg,center,terrain+offset[2]+cfg['map']['vertical_offset_m'],allowed)
    lift_route(base,dsm,bounds,offset,cfg)
    base_rot=make_poses(base,cfg)
    hits=frustum_hits(base-offset,base_rot,floor,cfg,dsm,bounds)
    coverage_scale,center_fit=fit_map_scale(base[:,:2]-offset[:2],hits,center-offset[:2],safe_allowed,cfg['map']['margin_m'])
    planned=base.copy()
    planned[:,:2]=offset[:2]+center_fit+coverage_scale*(base[:,:2]-center)
    roof,lift=lift_route(planned,dsm,bounds,offset,cfg)
    planned_rot=make_poses(planned,cfg)
    local_hits=frustum_hits(planned-offset,planned_rot,floor,cfg,dsm,bounds)
    points=local_hits.reshape(-1,2)
    full_bbox=np.r_[points.min(0)-cfg['map']['margin_m'],points.max(0)+cfg['map']['margin_m']]
    shift_lo,shift_hi=safe_allowed[:2]-full_bbox[:2],safe_allowed[2:]-full_bbox[2:]
    if np.any(shift_lo>shift_hi+1e-5):
        current_xy=planned[:,:2]-offset[:2]
        coverage_scale,center_fit=fit_map_scale(current_xy,local_hits,center_fit,safe_allowed,cfg['map']['margin_m'])
        planned[:,:2]=offset[:2]+center_fit+coverage_scale*(current_xy-center_fit)
        roof,extra_lift=lift_route(planned,dsm,bounds,offset,cfg);lift+=extra_lift
        if lift>cfg['trajectory']['max_auto_lift_m']: raise ValueError(f'DSM requires {lift:.2f} m lift, above max_auto_lift_m')
        planned_rot=make_poses(planned,cfg)
        points=frustum_hits(planned-offset,planned_rot,floor,cfg,dsm,bounds).reshape(-1,2)
        full_bbox=np.r_[points.min(0)-cfg['map']['margin_m'],points.max(0)+cfg['map']['margin_m']]
        shift_lo,shift_hi=safe_allowed[:2]-full_bbox[:2],safe_allowed[2:]-full_bbox[2:]
    if np.any(shift_lo>shift_hi+1e-5): raise ValueError(f'Camera footprint exceeds DOP: {full_bbox}, available {allowed}')
    shift=np.clip(np.zeros(2),shift_lo,np.maximum(shift_hi,shift_lo))
    planned[:,:2]+=shift
    center_fit+=shift
    roof,extra_lift=lift_route(planned,dsm,bounds,offset,cfg)
    lift+=extra_lift
    if lift>cfg['trajectory']['max_auto_lift_m']: raise ValueError(f'DSM requires {lift:.2f} m lift, above max_auto_lift_m')
    planned_rot=make_poses(planned,cfg)
    bbox=frustum_bbox(planned-offset,planned_rot,floor,cfg,dsm,bounds)
    if np.any(bbox[:2]<allowed[:2]-1e-4) or np.any(bbox[2:]>allowed[2:]+1e-4):
        raise ValueError(f'Planned frustum exceeds DOP after direct fit: {bbox}, available {allowed}')
    center_delta=center_fit-mid
    placement.update(offset_xy_m=center_delta.tolist(),randomized=bool(np.linalg.norm(center_delta)>0.01))
    atomic_json(placement_path,placement)
    cfg['trajectory'].pop('passes_override',None)
    speed=np.linalg.norm(np.diff(planned,axis=0),axis=1)*cfg['sequence']['fps']
    if len(speed) and speed.max()>cfg['trajectory']['max_speed_mps']*1.01:
        raise ValueError(f'Trajectory speed exceeds limit: {speed.max():.2f} m/s')
    travel=float(np.linalg.norm(np.diff(planned,axis=0),axis=1).sum())
    agl_full=planned[:,2]-roof
    if travel<cfg['trajectory']['min_distance_m']:
        raise ValueError(f'Auto route too short: {travel:.1f} m')
    if cfg['trajectory']['height_variation_ratio']>0 and agl_full.max()/agl_full.min()<cfg['trajectory']['min_scale_ratio']:
        raise ValueError('Auto route has insufficient height/scale variation')
    pos=planned[:frames];rot=planned_rot[:frames]
    cfg['sequence']['frames']=frames
    scene.mkdir(parents=True,exist_ok=True)
    doc=export_ges(scene/'trajectory.json',pos,rot,cfg)
    loader=PoseLoader(str(scene/'trajectory.json'),utm_zone=cfg['map']['utm_zone'])
    readback=loader.get_all_positions_utm()
    if np.max(np.linalg.norm(readback-pos,axis=1))>.02: raise ValueError('PoseLoader position round-trip failed')
    for i in np.linspace(0,frames-1,min(frames,50),dtype=int):
        if Rotation.from_matrix(loader.poses[i].rotation_matrix @ rot[i].T).magnitude()>1e-5:
            raise ValueError(f'PoseLoader orientation round-trip failed at frame {i}')
    local=pos-offset
    # The collision mesh and DSM in the reference scene use the same utm_offset-local frame.
    visibility=mesh_visibility(local,rot,cfg,ref/'mesh.npz')
    atomic_json(scene/'mesh_visibility.json',visibility)
    collision=validate_collision(local,dsm,bounds,ref/'mesh.npz',cfg['trajectory']['safe_distance_m'])
    collision['global_dsm_lift_m']=lift
    collision['adjusted_segments']=['all'] if lift>0 else []
    atomic_json(scene/'collision_report.json',collision)
    # Conservative full-frame corner intersection with lowest local terrain elevation.
    bbox=frustum_bbox(local,rot,floor,cfg,dsm,bounds)
    vf=np.deg2rad(cfg['sequence']['fov_vertical_deg'])
    image=cv2.imread(str(ref/dop_info['file']))
    if image is None: raise FileNotFoundError(ref/dop_info['file'])
    H,W=image.shape[:2]
    x0=int(np.floor((bbox[0]-db[0])/(db[2]-db[0])*W)); x1=int(np.ceil((bbox[2]-db[0])/(db[2]-db[0])*W))
    y0=int(np.floor((db[3]-bbox[3])/(db[3]-db[1])*H)); y1=int(np.ceil((db[3]-bbox[1])/(db[3]-db[1])*H))
    if min(x0,y0)<0 or x1>W or y1>H: raise ValueError('DOP crop exceeds reference image')
    crop=image[y0:y1,x0:x1]
    factor=max(1,int(np.ceil(max(crop.shape[:2])/cfg['map']['max_dop_size'])))
    if factor>1: crop=cv2.resize(crop,(int(np.ceil(crop.shape[1]/factor)),int(np.ceil(crop.shape[0]/factor))),interpolation=cv2.INTER_AREA)
    (scene/'dop').mkdir(exist_ok=True); cv2.imwrite(str(scene/'dop'/f'{cfg["map"]["dop_year"]}.jpg'),crop)
    # Keep the validated shared DSM without modifying its georeferencing.
    shutil.copy2(ref/'dsm.npz',scene/'dsm.npz')
    intr={'width':cfg['sequence']['width'],'height':cfg['sequence']['height'],'fy':cfg['sequence']['height']/(2*np.tan(vf/2))}
    intr.update(fx=intr['fy'],cx=intr['width']/2,cy=intr['height']/2)
    atomic_json(scene/'intrinsics.json',intr)
    scene_meta={'name':cfg['sequence']['name'],'width':cfg['sequence']['width'],'height':cfg['sequence']['height'],'num_frames':frames,'frame_rate':cfg['sequence']['fps'],'utm_zone':cfg['map']['utm_zone'],'utm_offset':offset.tolist(),'dsm':meta['dsm'],'dops':{f'dop_{cfg["map"]["dop_year"]}':{'file':f'dop/{cfg["map"]["dop_year"]}.jpg','bounds':bbox.tolist(),'gsd':(bbox[2]-bbox[0])/crop.shape[1]}},'vertical_offset_m':cfg['map']['vertical_offset_m'],'trajectory_center_offset_xy_m':center_delta.tolist()}
    atomic_json(scene/'meta.json',scene_meta)
    preview=scene/'preview';preview.mkdir(exist_ok=True)
    agl=local[:,2]-dsm_heights(dsm,bounds,local[:,:2])
    distance=float(np.linalg.norm(np.diff(local,axis=0),axis=1).sum())
    atomic_json(scene/'plan_report.json',challenge_report(cfg,local,rot,agl,float(coverage_scale)))
    atomic_json(scene/'config_used.json',cfg)
    fig,ax=plt.subplots(figsize=(7,7))
    ax.imshow(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),extent=bbox[[0,2,1,3]],origin='upper')
    segments=np.stack((local[:-1,:2],local[1:,:2]),axis=1)
    colored=LineCollection(segments,cmap='turbo',linewidth=1.6)
    colored.set_array((agl[:-1]+agl[1:])/2);ax.add_collection(colored)
    fig.colorbar(colored,ax=ax,label='Height above DSM (m)')
    ax.set(xlabel='Local X (m)',ylabel='Local Y (m)',title=f'Trajectory on DOP | {distance/1000:.2f} km | AGL {agl.min():.0f}-{agl.max():.0f} m')
    fig.tight_layout()
    fig.savefig(preview/'trajectory_on_dop.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection='3d')
    ax.plot(local[:,0],local[:,1],local[:,2],linewidth=.8)
    ref_height=cfg['trajectory']['reference_agl_m']
    ax.set_zlim(terrain+.5*ref_height,terrain+1.5*ref_height)
    ax.set(xlabel='Local X (m)',ylabel='Local Y (m)',zlabel='Local Z (m)')
    ax.set_title(f"{cfg['sequence']['name']} | base AGL {cfg['trajectory']['height_agl_m']:.0f} m")
    fig.savefig(preview/'trajectory_3d.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3));ax.plot(np.arange(frames)/cfg['sequence']['fps'],agl)
    ax.set(xlabel='Time (s)',ylabel='Height above DSM (m)');fig.tight_layout()
    fig.savefig(preview/'altitude_profile.png',dpi=150);plt.close(fig)
    atomic_json(preview/'summary.json',{'frames':frames,'distance_m':distance,'agl_min_m':float(agl.min()),'agl_max_m':float(agl.max()),'bbox_local':bbox.tolist(),'collision':collision})
    with (scene/'poses.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['frame_id','x','y','z','qw','qx','qy','qz','latitude','longitude','altitude','fov_vertical'])
        for i,p in enumerate(loader.poses):
            c=doc['cameraFrames'][i]['coordinate'];writer.writerow([i,p.x-offset[0],p.y-offset[1],p.z-offset[2],p.qw,p.qx,p.qy,p.qz,c['latitude'],c['longitude'],c['altitude']-offset[2],doc['cameraFrames'][i]['fovVertical']])
    return doc


def local_colored_model(cfg):
    """Use the downloaded MovingDrone mesh, colored by its DOP; official renderer handles images."""
    import open3d as o3d
    ref=Path(cfg['map']['reference_scene'])
    meta=json.loads((ref/'meta.json').read_text())
    data=np.load(ref/'mesh.npz')
    vertices=data['vertices'].astype(np.float64)
    faces=data['faces'].astype(np.int32)
    dop=meta['dops'][f'dop_{cfg["map"]["dop_year"]}']
    image=cv2.imread(str(ref/dop['file']))
    if image is None: raise ValueError('Reference DOP missing')
    bounds=np.asarray(dop['bounds']); h,w=image.shape[:2]
    col=np.clip(((vertices[:,0]-bounds[0])/(bounds[2]-bounds[0])*w).astype(int),0,w-1)
    row=np.clip(((bounds[3]-vertices[:,1])/(bounds[3]-bounds[1])*h).astype(int),0,h-1)
    colors=image[row,col,::-1].copy()/255.0
    uv=np.column_stack(((vertices[:,0]-bounds[0])/(bounds[2]-bounds[0]),
                        (vertices[:,1]-bounds[1])/(bounds[3]-bounds[1])))
    vertices+=np.asarray(meta['utm_offset'])
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors=o3d.utility.Vector3dVector(colors)
    mesh.triangle_uvs=o3d.utility.Vector2dVector(np.clip(uv[faces].reshape(-1,2),0,1))
    mesh.compute_vertex_normals()
    material=o3d.visualization.rendering.MaterialRecord();material.shader='defaultUnlit'
    material.albedo_img=o3d.geometry.Image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    model=o3d.visualization.rendering.TriangleMeshModel()
    model.materials=[material]
    model.meshes=[o3d.visualization.rendering.TriangleMeshModel.MeshInfo(mesh,'movingdrone_mesh',0)]
    return [('movingdrone_mesh',model)]


def render(cfg,scene,doc):
    from dataset.create_movingdrone import render_sequence,encode_video_from_frames
    n=len(doc['cameraFrames']); size=cfg['render']['chunk_frames']
    work=scene/'_chunks';work.mkdir(exist_ok=True)
    videos=[]
    renderer_source=(ROOT/'dataset/create_movingdrone.py').read_bytes()+Path(__file__).read_bytes()
    digest=hashlib.sha256(json.dumps(doc,sort_keys=True).encode()+json.dumps(cfg,sort_keys=True).encode()+renderer_source).hexdigest()
    for start in range(0,n,size):
        end=min(n,start+size); chunk=work/f'chunk_{start//size:04d}';chunk.mkdir(exist_ok=True)
        state=chunk/'state.json'; video=chunk/'video.mp4'; path=chunk/'trajectory.json'
        chunk_doc=dict(doc,cameraFrames=doc['cameraFrames'][start:end],numFrames=end-start-1)
        signature=hashlib.sha256((digest+str(start)+str(end)).encode()).hexdigest()
        if state.exists() and video.exists() and path.exists() and json.loads(state.read_text()).get('signature')==signature and json.loads(path.read_text())==chunk_doc:
            try: check_video(video,end-start,cfg); videos.append(video);continue
            except ValueError: pass
        path.write_text(json.dumps(chunk_doc))
        models=local_colored_model(cfg)
        render_sequence(models,str(path),str(chunk),utm_zone=cfg['map']['utm_zone'],save_depth=False,save_normals=False,realism=False)
        frames=sorted((chunk/'rendering').glob('frame_*.jpg'))
        if len(frames)!=end-start: raise ValueError(f'Chunk {start//size}: {len(frames)} rendered frames; expected {end-start}')
        if not encode_video_from_frames(chunk/'rendering',video,cfg['sequence']['fps'],cfg['sequence']['width'],cfg['sequence']['height']): raise ValueError('Video encode failed')
        check_video(video,end-start,cfg)
        atomic_json(state,{'chunk_id':start//size,'start_frame':start,'end_frame':end,'expected_frames':end-start,'status':'completed','signature':signature})
        videos.append(video)
    manifest=work/'concat.txt';manifest.write_text(''.join(f"file '{p.resolve()}'\n" for p in videos))
    target=scene/'video.mp4'; subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',str(manifest),'-c','copy',str(target)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    check_video(target,n,cfg)
    cap=cv2.VideoCapture(str(target)); samples=[]
    for frame in np.linspace(0,n-1,min(9,n),dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,image=cap.read()
        if not ok: raise ValueError(f'Cannot decode preview frame {frame}')
        samples.append(cv2.resize(image,(320,180)))
    cap.release()
    rows=[np.hstack(samples[i:i+3]) for i in range(0,len(samples),3)]
    cv2.imwrite(str(scene/'preview'/'camera_view_samples.jpg'),np.vstack(rows))


def generate_depths(args,cfg):
    from tqdm import tqdm
    root=Path(cfg['output_root']); variants=cfg['ablations']
    scenes=[args.scene] if args.scene else sorted(p.name for p in root.iterdir() if (p/'reference'/'meta.json').is_file())
    names=[args.variant] if args.variant else list(variants)
    jobs=[(s,v) for s in scenes for v in names if (root/s/v/'trajectory.json').is_file()]
    manifest_path=root/'depth_generation.json'
    manifest={'status':'running','scenes':scenes,'variants':names,'frames_per_variant':{},'jobs':[]}
    chunk_size=cfg['render']['chunk_frames']
    source=ROOT/'dataset/create_movingdrone.py'
    atomic_json(manifest_path,manifest)
    with tqdm(jobs,desc='Depth maps',unit='variant') as bar:
        for scene_name,variant in bar:
            scene=root/scene_name/variant
            variant_cfg=json.loads((scene/'config_used.json').read_text())
            variant_cfg['map']['reference_scene']=str(root/scene_name/'reference')
            doc=json.loads((scene/'trajectory.json').read_text()); n=len(doc['cameraFrames'])
            h,w=doc['height'],doc['width'];depth_dir=scene/'depth';depth_dir.mkdir(exist_ok=True)
            mesh=Path(variant_cfg['map']['reference_scene'])/'mesh.npz'
            signature=hashlib.sha256((scene/'trajectory.json').read_bytes()+source.read_bytes()+str(mesh.stat().st_mtime_ns).encode()).hexdigest()
            work=scene/'_depth_work';work.mkdir(exist_ok=True)
            for start in range(0,n,chunk_size):
                end=min(n,start+chunk_size);chunk=work/f'chunk_{start//chunk_size:04d}'
                state=chunk/'state.json';expected=[depth_dir/f'depth_{i:04d}.png' for i in range(start,end)]
                if state.exists() and all(p.is_file() and p.with_suffix('.json').is_file() for p in expected) and json.loads(state.read_text()).get('signature')==signature:
                    continue
                chunk.mkdir(exist_ok=True)
                chunk_doc=dict(doc,cameraFrames=doc['cameraFrames'][start:end],numFrames=end-start-1)
                path=chunk/'trajectory.json';path.write_text(json.dumps(chunk_doc))
                from dataset.create_movingdrone import render_sequence
                render_sequence(local_colored_model(variant_cfg),str(path),str(chunk),utm_zone=variant_cfg['map']['utm_zone'],
                                save_depth=True,save_normals=False,save_rgb=False,depth_format='png16')
                files=sorted((chunk/'depth').glob('depth_*.png'))
                if len(files)!=end-start: raise ValueError(f'{scene_name}/{variant}: depth chunk {start}:{end} has {len(files)} frames')
                for i,src in enumerate(files,start):
                    dest=depth_dir/f'depth_{i:04d}.png';os.replace(src,dest);os.replace(src.with_suffix('.json'),dest.with_suffix('.json'))
                atomic_json(state,{'start_frame':start,'end_frame':end,'status':'completed','signature':signature})
                bar.set_postfix_str(f'{scene_name}/{variant}: {end}/{n}')
            if len(list(depth_dir.glob('depth_*.png')))!=n: raise ValueError(f'{scene_name}/{variant}: incomplete depth output')
            atomic_json(depth_dir/'metadata.json',{'frames':n,'width':w,'height':h,'units':'metres','representation':'ray_distance','dtype':'uint16','format':'png+per-frame-json','mesh':'reference/mesh.npz'})
            manifest['frames_per_variant'][f'{scene_name}/{variant}']=n
            manifest['jobs'].append({'scene':scene_name,'variant':variant,'status':'done'})
            atomic_json(manifest_path,manifest)
    manifest['status']='completed';atomic_json(manifest_path,manifest)
    print(f'Depth maps complete: {manifest_path}')


def create_ablation_summary(sequence, variants):
    paths=[sequence/name for name in variants]
    files=[p/'video.mp4' for p in paths]
    previews=[p/'preview' for p in paths]
    required=('trajectory_3d.png','trajectory_on_dop.png')
    if not all(f.is_file() for f in files) or not all((p/name).is_file() for p in previews for name in required):
        return False

    reference=sequence/'reference'
    meta=json.loads((reference/'meta.json').read_text())
    year=max(meta['dops'],key=lambda key:int(key.removeprefix('dop_')))
    dop=meta['dops'][year]
    image=cv2.imread(str(reference/dop['file']))
    if image is None:
        raise FileNotFoundError(reference/dop['file'])
    bounds=np.asarray(dop['bounds'],float)
    tracks=[]
    for path in paths:
        with (path/'poses.csv').open() as f:
            tracks.append(np.asarray([[float(row[k]) for k in ('x','y','z')] for row in csv.DictReader(f)]))
    xy=np.concatenate([track[:,:2] for track in tracks])
    margin=np.maximum(25.,.08*np.ptp(xy,axis=0))
    crop_bounds=np.r_[xy.min(axis=0)-margin,xy.max(axis=0)+margin]
    crop_bounds[[0,2]]=np.clip(crop_bounds[[0,2]],bounds[0],bounds[2])
    crop_bounds[[1,3]]=np.clip(crop_bounds[[1,3]],bounds[1],bounds[3])
    h,w=image.shape[:2]
    x0=int(np.floor((crop_bounds[0]-bounds[0])/(bounds[2]-bounds[0])*w))
    x1=int(np.ceil((crop_bounds[2]-bounds[0])/(bounds[2]-bounds[0])*w))
    y0=int(np.floor((bounds[3]-crop_bounds[3])/(bounds[3]-bounds[1])*h))
    y1=int(np.ceil((bounds[3]-crop_bounds[1])/(bounds[3]-bounds[1])*h))
    crop=image[y0:y1,x0:x1]
    crop_meta={'year':year,'bounds':crop_bounds.tolist(),
        'trajectory_bounds':[xy.min(axis=0).tolist(),xy.max(axis=0).tolist()],
        'margin_xy_m':margin.tolist()}
    crop_name=f'{year.removeprefix("dop_")}_cropped.jpg'
    with np.load(reference/'dsm.npz') as data:
        dsm=data['height'];dsm_bounds=data['bounds']
    for path,track in zip(paths,tracks):
        dop_dir=path/'dop';dop_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(dop_dir/crop_name),crop)
        atomic_json(dop_dir/f'{year.removeprefix("dop_")}_cropped.json',crop_meta)
        for old in dop_dir.glob('ablation_crop.*'):
            old.unlink(missing_ok=True)
        agl=track[:,2]-dsm_heights(dsm,dsm_bounds,track[:,:2])
        segments=np.stack((track[:-1,:2],track[1:,:2]),axis=1)
        colored=LineCollection(segments,cmap='turbo',linewidth=1.6)
        colored.set_array((agl[:-1]+agl[1:])/2)
        fig,ax=plt.subplots(figsize=(7,7))
        ax.imshow(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),
                  extent=crop_bounds[[0,2,1,3]],origin='upper')
        ax.add_collection(colored)
        fig.colorbar(colored,ax=ax,label='Height above DSM (m)')
        distance=np.linalg.norm(np.diff(track[:,:2],axis=0),axis=1).sum()
        ax.set(xlabel='Local X (m)',ylabel='Local Y (m)',
               title=f'Trajectory on DOP | {distance/1000:.2f} km | AGL {agl.min():.0f}-{agl.max():.0f} m')
        fig.tight_layout()
        fig.savefig(path/'preview'/'trajectory_on_dop.png',dpi=150)
        plt.close(fig)
    for old in (sequence/'dop'/'ablation_crop.jpg',sequence/'dop'/'ablation_crop.json'):
        old.unlink(missing_ok=True)
    try:
        (sequence/'dop').rmdir()
    except OSError:
        pass

    card_w,card_h,gap=480,240,8
    canvas=np.full((card_h*2+gap,card_w*3+gap*2,3),255,np.uint8)
    for i,(name,preview) in enumerate(zip(variants,previews)):
        row,col=divmod(i,3);x=col*(card_w+gap);y=row*(card_h+gap)
        cv2.rectangle(canvas,(x,y),(x+card_w-1,y+card_h-1),(210,215,220),1)
        cv2.putText(canvas,name,(x+9,y+20),cv2.FONT_HERSHEY_SIMPLEX,.55,(35,35,35),1,cv2.LINE_AA)
        for j,filename in enumerate(required):
            panel=cv2.imread(str(preview/filename))
            if j==1:
                panel=panel[:,:int(panel.shape[1]*.80)]
            gray=cv2.cvtColor(panel,cv2.COLOR_BGR2GRAY)
            yy,xx=np.where(gray<247)
            if len(xx):
                x0=max(0,int(xx.min())-8);x1=min(panel.shape[1],int(xx.max())+9)
                y0=max(0,int(yy.min())-8);y1=min(panel.shape[0],int(yy.max())+9)
                panel=panel[y0:y1,x0:x1]
            max_w,max_h=224,178
            scale=min(max_w/panel.shape[1],max_h/panel.shape[0])
            size=(max(1,int(panel.shape[1]*scale)),max(1,int(panel.shape[0]*scale)))
            panel=cv2.resize(panel,size,interpolation=cv2.INTER_AREA)
            px=x+8+j*232;py=y+34
            cv2.putText(canvas,'3D' if j==0 else 'DOP',(px,py+11),cv2.FONT_HERSHEY_SIMPLEX,.38,(90,90,90),1,cv2.LINE_AA)
            ox=px+(max_w-size[0])//2;oy=py+15+(max_h-size[1])//2
            canvas[oy:oy+size[1],ox:ox+size[0]]=panel
    cv2.imwrite(str(sequence/'ablation_comparison.png'),canvas)
    return True


def generate_all(args, cfg):
    from tqdm import tqdm
    root=Path(cfg['output_root'])
    variants=list(cfg['ablations'])
    scenes=sorted(p.name for p in root.iterdir() if (p/'reference'/'meta.json').is_file())
    if args.scene:
        scenes=[name for name in scenes if name==args.scene]
    if not scenes:
        raise ValueError(f'No reference scenes found under {root}')
    run_seed=args.seed if args.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    log_path=root/f'generation_{run_seed}.log'
    manifest_path=root/f'generation_{run_seed}.json'
    manifest={'seed':run_seed,'scenes':scenes,'variants':variants,'status':'running','jobs':[]}
    atomic_json(manifest_path,manifest)
    with log_path.open('a') as log, tqdm(total=len(scenes)*len(variants),desc='MovingDrone',unit='variant') as bar:
        for name in scenes:
            seq_seed=int.from_bytes(hashlib.sha256(f'{run_seed}:{name}'.encode()).digest()[:4],'little')
            sequence=root/name
            (sequence/'ablation_comparison.png').unlink(missing_ok=True)
            for variant in variants:
                target=sequence/variant
                target.mkdir(parents=True,exist_ok=True)
                for filename in ('video.mp4','poses.csv','trajectory.json','meta.json','intrinsics.json',
                                 'config_used.json','plan_report.json','collision_report.json','mesh_visibility.json','dsm.npz'):
                    (target/filename).unlink(missing_ok=True)
                for filename in ('trajectory_3d.png','trajectory_on_dop.png','altitude_profile.png',
                                 'camera_view_samples.jpg','summary.json'):
                    (target/'preview'/filename).unlink(missing_ok=True)
                for filename in ('ablation_crop.jpg','ablation_crop.json'):
                    (target/'dop'/filename).unlink(missing_ok=True)
            for variant in variants:
                target=sequence/variant
                command=[sys.executable,str(Path(__file__).resolve()),'generate','--config',str(args.config.resolve()),
                         '--scene',name,'--variant',variant,'--seed',str(seq_seed),'--keep-variant-name']
                if args.fov_scale is not None: command.extend(['--fov-scale',str(args.fov_scale)])
                if args.tilt_deg is not None: command.extend(['--tilt-deg',str(args.tilt_deg)])
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode==0:
                    check=[sys.executable,str(Path(__file__).resolve()),'validate','--config',str(args.config.resolve()),
                           '--scene',name,'--variant',variant]
                    result=subprocess.run(check,stdout=log,stderr=subprocess.STDOUT)
                status='done' if result.returncode==0 else 'failed'
                manifest['jobs'].append({'scene':name,'variant':variant,'seed':seq_seed,'status':status})
                atomic_json(manifest_path,manifest)
                bar.set_postfix_str(f'{name}/{variant}: {status}')
                bar.update(1)
    manifest['status']='completed' if all(j['status']=='done' for j in manifest['jobs']) else 'completed_with_failures'
    atomic_json(manifest_path,manifest)
    print(f'Batch {manifest["status"]}: {manifest_path}; details: {log_path}')


def generate_failed(args, cfg):
    from tqdm import tqdm
    manifest=json.loads(args.manifest.read_text())
    failed=[job for job in manifest['jobs'] if job['status']!='done' and (not args.scene or job['scene']==args.scene)]
    if not failed:
        print('No failed variants to retry');return
    log_path=args.manifest.with_name(args.manifest.stem+'_retry.log')
    with log_path.open('a') as log, tqdm(failed,desc='Retrying',unit='variant') as bar:
        for job in bar:
            target=Path(cfg['output_root'])/job['scene']/job['variant']
            for filename in ('video.mp4','poses.csv','trajectory.json','meta.json','intrinsics.json',
                             'config_used.json','plan_report.json','collision_report.json','mesh_visibility.json','dsm.npz'):
                (target/filename).unlink(missing_ok=True)
            for filename in ('trajectory_3d.png','trajectory_on_dop.png','altitude_profile.png',
                             'camera_view_samples.jpg','summary.json'):
                (target/'preview'/filename).unlink(missing_ok=True)
            for filename in ('ablation_crop.jpg','ablation_crop.json'):
                (target/'dop'/filename).unlink(missing_ok=True)
            command=[sys.executable,str(Path(__file__).resolve()),'generate','--config',str(args.config.resolve()),
                     '--scene',job['scene'],'--variant',job['variant'],'--seed',str(job['seed']),'--keep-variant-name']
            if args.fov_scale is not None: command.extend(['--fov-scale',str(args.fov_scale)])
            if args.tilt_deg is not None: command.extend(['--tilt-deg',str(args.tilt_deg)])
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:
                check=[sys.executable,str(Path(__file__).resolve()),'validate','--config',str(args.config.resolve()),
                       '--scene',job['scene'],'--variant',job['variant'],'--seed',str(job['seed']),'--keep-variant-name']
                if args.fov_scale is not None: check.extend(['--fov-scale',str(args.fov_scale)])
                if args.tilt_deg is not None: check.extend(['--tilt-deg',str(args.tilt_deg)])
                result=subprocess.run(check,stdout=log,stderr=subprocess.STDOUT)
            job['status']='done' if result.returncode==0 else 'failed'
            job['retry_log']=str(log_path)
            atomic_json(args.manifest,manifest)
            bar.set_postfix_str(f'{job["scene"]}/{job["variant"]}: {job["status"]}')
    manifest['status']='completed' if all(j['status']=='done' for j in manifest['jobs']) else 'completed_with_failures'
    atomic_json(args.manifest,manifest)
    print(f'Batch {manifest["status"]}: {args.manifest}; details: {log_path}')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','generate','generate-all','generate-failed','validate','preview','depth']);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'));p.add_argument('--scene');p.add_argument('--variant');p.add_argument('--frames',type=int);p.add_argument('--manifest',type=Path);p.add_argument('--fov-scale',type=float);p.add_argument('--tilt-deg',type=float)
    p.add_argument('--seed',type=int);p.add_argument('--keep-variant-name',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if args.command=='generate-all':
        generate_all(args,cfg)
        return
    if args.command=='generate-failed':
        if not args.manifest: p.error('generate-failed requires --manifest')
        generate_failed(args,cfg)
        return
    if args.command=='depth':
        generate_depths(args,cfg)
        return
    if args.variant is None: args.variant='baseline'
    if args.scene: cfg['map']['reference_scene']=str(Path(cfg['output_root'])/args.scene/'reference')
    reference=Path(cfg['map']['reference_scene'])
    with np.load(reference/'dsm.npz') as data:
        ref_agl,ref_fov=reference_parameters(reference,data['height'],data['bounds'])
    variants=cfg.pop('ablations')
    if args.variant not in variants: p.error(f'Unknown variant {args.variant}; choose from {", ".join(variants)}')
    cfg['trajectory'].update(variants[args.variant])
    if args.tilt_deg is not None: cfg['trajectory']['tilt_from_nadir_deg']=args.tilt_deg
    fov_scale=args.fov_scale if args.fov_scale is not None else cfg['sequence'].get('fov_scale',1.)
    if not 0<fov_scale<=1: p.error('--fov-scale must be in (0, 1]')
    cfg['sequence']['fov_scale']=fov_scale
    cfg['sequence']['fov_vertical_deg']=ref_fov*fov_scale
    cfg['trajectory']['reference_agl_m']=ref_agl
    cfg['trajectory']['height_agl_m']=ref_agl*(1+cfg['trajectory']['height_offset_ratio'])
    cfg['trajectory']['height_variation_m']=ref_agl*cfg['trajectory']['height_variation_ratio']
    cfg['trajectory']['max_agl_m']=ref_agl*cfg['trajectory']['max_agl_ratio']
    source_name=Path(cfg['map']['reference_scene']).parent.name
    variant_name=args.variant
    if args.seed is not None:
        cfg['trajectory']['seed']=args.seed
        if not args.keep_variant_name: variant_name+=f'_seed{args.seed}'
    n=args.frames or cfg['sequence']['frames']
    cfg['sequence']['name']=f'{source_name}_{variant_name}'
    scene=Path(cfg['output_root'])/source_name/variant_name
    if args.command=='preview':
        if create_ablation_summary(Path(cfg['output_root'])/source_name,list(variants)):
            print(f'Created ablation preview: {Path(cfg["output_root"])/source_name}')
        else:
            print(f'Waiting for all ablations: {Path(cfg["output_root"])/source_name}')
        return
    if args.command=='validate':
        check_video(scene/'video.mp4',n,cfg)
        with (scene/'poses.csv').open() as f: rows=list(csv.DictReader(f))
        if len(rows)!=n: raise ValueError('Pose row count mismatch')
        ids=[int(row['frame_id']) for row in rows]
        if ids!=list(range(n)): raise ValueError('Frame IDs are not contiguous')
        positions=np.array([[float(row[k]) for k in ('x','y','z')] for row in rows])
        if not np.isfinite(positions).all(): raise ValueError('Non-finite pose position')
        speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)*cfg['sequence']['fps']
        if len(speed) and speed.max()>cfg['trajectory']['max_speed_mps']*1.01: raise ValueError(f'Speed discontinuity: {speed.max():.2f} m/s')
        meta=json.loads((scene/'meta.json').read_text())
        if len(meta['utm_offset'])!=3 or len(meta['dops'])!=1: raise ValueError('Invalid shared map metadata')
        if not (scene/'collision_report.json').exists() or not (scene/'mesh_visibility.json').exists(): raise ValueError('Missing geometry validation report')
        plan=json.loads((scene/'plan_report.json').read_text())
        if plan['frames']!=n or plan['agl_min_m']<cfg['trajectory']['min_agl_m']-.1 or plan['agl_max_m']>cfg['trajectory']['max_agl_m']+.1:
            raise ValueError('Plan report violates frame or AGL constraints')
        if n==cfg['sequence']['frames'] and (plan['distance_m']<cfg['trajectory']['min_distance_m'] or (cfg['trajectory']['height_variation_ratio']>0 and plan['scale_ratio']<cfg['trajectory']['min_scale_ratio'])):
            raise ValueError('Plan report violates distance or scale constraints')
        depth=scene/'depth';pngs=sorted(depth.glob('depth_*.png')) if depth.exists() else []
        npzs=list(depth.glob('depth_*.npz')) if depth.exists() else []
        if depth.exists() and len(pngs)+len(npzs)!=n: raise ValueError('Depth frame count mismatch')
        if pngs:
            if any(not p.with_suffix('.json').is_file() for p in pngs): raise ValueError('Depth PNG metadata missing')
            for p in (pngs[0],pngs[len(pngs)//2],pngs[-1]):
                image=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
                if image is None or image.shape!=(cfg['sequence']['height'],cfg['sequence']['width']) or image.dtype!=np.uint16:
                    raise ValueError(f'Invalid depth PNG: {p}')
        if (scene/'normals').exists(): raise ValueError('Unexpected normals output exists')
        print(f'Validated {n} frames: {scene}');return
    doc=prepare(cfg,scene,n)
    if args.command=='plan':
        print(f'Planned {n} frames without rendering: {scene}');return
    render(cfg,scene,doc)
    if args.command=='generate' and create_ablation_summary(Path(cfg['output_root'])/source_name,list(variants)):
        print(f'Created ablation preview: {Path(cfg["output_root"])/source_name}')
    print(f'Complete: {scene}')


if __name__=='__main__': main()
