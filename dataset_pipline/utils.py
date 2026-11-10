"""Shared file, image, and camera-geometry helpers for the tile pipeline."""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def compact_output(scene):
    cfg=yaml.safe_load((scene/'config.yaml').read_text())
    ref=Path(cfg['map']['reference_scene'])
    for path,target in ((scene/'dsm.npz',ref/'dsm.npz'),
                        (scene/'dop'/f'{cfg["map"]["dop_year"]}.jpg',scene/'dop_view.jpg')):
        if target.is_file() and path.is_file() and not path.is_symlink():
            path.unlink();path.symlink_to(target.resolve())


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


def check_depth(scene,n,cfg):
    depth_dir=scene/'depth';files=sorted(depth_dir.glob('depth_[0-9][0-9][0-9][0-9][0-9][0-9].png'))
    if len(files)!=n: raise ValueError(f'Depth frame count mismatch: {len(files)} != {n}')
    meta=json.loads((depth_dir/'metadata.json').read_text())
    scale=float(meta.get('scale_m_per_unit',0))
    if meta.get('frames')!=n or not 0<scale<=1 or meta.get('depth_type')!='camera_z':
        raise ValueError(f'Invalid depth metadata: {depth_dir}/metadata.json')
    h,w=cfg['sequence']['height'],cfg['sequence']['width']
    for i in (0,n//2,n-1):
        image=cv2.imread(str(depth_dir/f'depth_{i:06d}.png'),cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype!=np.uint16 or image.shape!=(h,w):
            raise ValueError(f'Invalid depth image: {depth_dir}/depth_{i:06d}.png')


def crop_dop(image,source_bounds,target_bounds):
    h,w=image.shape[:2];x0,y0,x1,y1=source_bounds
    c0=int(np.floor((target_bounds[0]-x0)/(x1-x0)*w))
    c1=int(np.ceil((target_bounds[2]-x0)/(x1-x0)*w))
    r0=int(np.floor((y1-target_bounds[3])/(y1-y0)*h))
    r1=int(np.ceil((y1-target_bounds[1])/(y1-y0)*h))
    c0,c1=np.clip([c0,c1],0,w);r0,r1=np.clip([r0,r1],0,h)
    if c1<=c0 or r1<=r0: raise ValueError('UAV view produced an empty DOP crop')
    actual=np.array([x0+c0/w*(x1-x0),y1-r1/h*(y1-y0),
                     x0+c1/w*(x1-x0),y1-r0/h*(y1-y0)],float)
    return image[r0:r1,c0:c1],actual


def frustum_hits(local,rot,floor,cfg,dsm,bounds):
    vf=np.deg2rad(cfg['sequence']['fov_vertical_deg'])
    fy=1/np.tan(vf/2); fx=fy/(cfg['sequence']['width']/cfg['sequence']['height'])
    rays=np.array([[x/fx,y/fy,1] for x in np.linspace(-1,1,5) for y in np.linspace(-1,1,5)],float)
    directions=np.einsum('nij,kj->nki',rot,rays)
    ground=directions[:,:,2]<-.05
    directions[:,:,2]=np.where(ground,directions[:,:,2],-1.)
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
    xy=hits[:,:,:2]
    xy[~ground]=np.nan
    return xy




