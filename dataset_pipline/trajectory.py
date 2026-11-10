"""Manual-route camera orientations and GES-compatible pose export."""
import json
from pathlib import Path
import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation


def make_forward_poses(positions, cfg):
    plan=cfg['trajectory']
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1))]
    look=float(plan['heading_lookahead_m'])
    if look <= 0: raise ValueError('heading_lookahead_m must be positive')
    ahead=np.minimum(arc+look,arc[-1]); behind=np.maximum(arc-look,0)
    front=np.column_stack([np.interp(ahead,arc,positions[:,j]) for j in range(2)])
    back=np.column_stack([np.interp(behind,arc,positions[:,j]) for j in range(2)])
    direction=front-positions[:,:2]
    terminal=np.linalg.norm(direction,axis=1)<1e-8
    direction[terminal]=positions[terminal,:2]-back[terminal]
    heading=np.unwrap(np.arctan2(direction[:,1],direction[:,0]))
    direction=np.column_stack((np.cos(heading),np.sin(heading)))
    center=np.asarray(plan['look_at_center_xy'],float)
    toward_center=center-positions[:,:2]
    toward_center/=np.maximum(np.linalg.norm(toward_center,axis=1,keepdims=True),1e-9)
    weight=float(plan.get('camera_center_weight',.5))
    direction=(1-weight)*direction+weight*toward_center
    direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
    vertical=positions[:,2]-float(plan['look_at_z'])
    if np.any(vertical<=0): raise ValueError('Camera must stay above its look-at surface')
    angle=np.deg2rad(float(plan['camera_tilt_lock_deg']))
    horizontal=direction*(np.tan(angle)*vertical)[:,None]
    forward=np.column_stack((horizontal,-vertical))
    forward/=np.maximum(np.linalg.norm(forward,axis=1)[:,None],1e-9)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right,axis=1)[:,None]
    down=np.cross(forward,right)
    return np.stack((right,down,forward),axis=2)


def plan_altitude(ground, agl, clearance, max_vertical_speed, fps):
    altitude=np.maximum(ground+agl,ground+clearance)
    step=max_vertical_speed/fps
    for i in range(len(altitude)-2,-1,-1):
        altitude[i]=max(altitude[i],altitude[i+1]-step)
    for i in range(1,len(altitude)):
        altitude[i]=max(altitude[i],altitude[i-1]-step)
    return altitude


def export_ges(path, positions, rotations, cfg):
    zone = cfg['map']['utm_zone']
    to_geo = Transformer.from_crs(f'EPSG:326{zone}', 'EPSG:4979', always_xy=True)
    to_ecef = Transformer.from_crs('EPSG:4979','EPSG:4978',always_xy=True)
    lon,lat,alt = to_geo.transform(positions[:,0],positions[:,1],positions[:,2])
    ex,ey,ez = to_ecef.transform(lon,lat,alt)
    frames=[]
    for i in range(len(positions)):
        la,lo=np.deg2rad(lat[i]),np.deg2rad(lon[i]); sl,cl=np.sin(la),np.cos(la); so,co=np.sin(lo),np.cos(lo)
        ecef_to_enu=np.array([[-so,co,0],[-sl*co,-sl*so,cl],[cl*co,cl*so,sl]])
        angles=Rotation.from_matrix(ecef_to_enu.T @ rotations[i]).as_euler('XYZ',degrees=True)
        frames.append({'coordinate':{'latitude':float(lat[i]),'longitude':float(lon[i]),'altitude':float(alt[i])},'position':{'x':float(ex[i]),'y':float(ey[i]),'z':float(ez[i])},'rotation':dict(zip('xyz',map(float,angles))),'fovVertical':cfg['sequence']['fov_vertical_deg']})
    doc={'name':cfg['sequence']['name'],'width':cfg['sequence']['width'],'height':cfg['sequence']['height'],'frameRate':cfg['sequence']['fps'],'numFrames':len(frames)-1,'durationSeconds':len(frames)/cfg['sequence']['fps'],'cameraFrames':frames}
    Path(path).write_text(json.dumps(doc))
    return doc
