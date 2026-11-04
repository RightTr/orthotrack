"""Deterministic metric trajectories and GES-compatible pose export."""
import json
from pathlib import Path
import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation


def make_poses(positions, cfg):
    return make_forward_poses(positions, cfg)


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
    side=np.deg2rad(plan['side_look_deg'])*np.sin(2*np.pi*arc/max(arc[-1],1))
    heading+=side
    fov=float(cfg['sequence']['fov_vertical_deg'])
    max_tilt=90-fov/2-1
    tilt=plan['tilt_from_nadir_deg']+plan['tilt_variation_deg']*np.sin(2*np.pi*arc/max(arc[-1],1))
    if np.any(tilt<=0) or np.any(tilt>=max_tilt):
        raise ValueError(f'Camera tilt must be between 0 and {max_tilt:.1f} degrees for this FOV')
    tilt=np.deg2rad(tilt)
    forward=np.column_stack((np.cos(heading)*np.sin(tilt),np.sin(heading)*np.sin(tilt),-np.cos(tilt)))
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right,axis=1)[:,None]
    down=np.cross(forward,right)
    return np.stack((right,down,forward),axis=2)


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
