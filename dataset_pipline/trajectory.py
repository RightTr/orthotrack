"""Deterministic metric trajectories and GES-compatible pose export."""
import json
from pathlib import Path
import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation


def make_poses(positions, cfg):
    u = np.linspace(0,1,len(positions))
    tilt = np.deg2rad(cfg['trajectory']['tilt_from_nadir_deg']
                     + cfg['trajectory']['tilt_variation_deg']*np.sin(4*np.pi*u))
    velocity = np.gradient(positions, axis=0)
    tangent = velocity.copy(); tangent[:,2] = 0
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:,None], 1e-9)
    side = np.deg2rad(cfg['trajectory']['side_look_deg'])*np.sin(6*np.pi*u)
    heading = np.arctan2(tangent[:,1],tangent[:,0])+side
    tangent[:,0],tangent[:,1] = np.cos(heading),np.sin(heading)
    forward = np.column_stack((tangent[:,0]*np.sin(tilt), tangent[:,1]*np.sin(tilt), -np.cos(tilt)))
    right = np.cross(forward, np.array([0.,0.,1.])); right /= np.linalg.norm(right,axis=1)[:,None]
    down = np.cross(forward,right)
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
