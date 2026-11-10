"""Turn a hand-drawn DOP path into a smooth, metric UAV route."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def preview_route(points, bounds, frames, fps, plan, saved_smoothing=None):
    points=np.asarray(points,float);bounds=np.asarray(bounds,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<4 or not np.isfinite(points).all():
        raise ValueError('Draw a continuous route with at least four finite points')
    if np.any(points<bounds[:2]) or np.any(points>bounds[2:]):
        raise ValueError('Drawn route leaves its DOP')
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>.25]
    points=points[keep]
    if len(points)<4: raise ValueError('Draw a longer route before rendering')
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    drawn_length=float(arc[-1])
    sample=np.linspace(0,drawn_length,max(2000,int(drawn_length/.5)))
    dense=np.column_stack([np.interp(sample,arc,points[:,j]) for j in range(2)])
    n=int(frames);fps=float(fps)
    if n<2 or fps<=0: raise ValueError('Invalid frame count or FPS')
    smooth_m=float(saved_smoothing if saved_smoothing is not None else plan.get('route_smoothing_m',8.))
    smooth=(dense if saved_smoothing is not None else
            np.column_stack([gaussian_filter1d(dense[:,j],smooth_m/(sample[1]-sample[0]),mode='nearest')
                             for j in range(2)]))
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(smooth,axis=0),axis=1))]
    tangent=np.unwrap(np.arctan2(np.gradient(smooth[:,1]),np.gradient(smooth[:,0])))
    curvature=np.abs(np.gradient(tangent,distance))
    slow_distance=float(plan.get('turn_slowdown_distance_m',25.))
    factor=gaussian_filter1d(np.clip(1/(1+curvature*slow_distance),.08,1.),
                             max(1.,3./(sample[1]-sample[0])))
    segment_factor=np.maximum(.08,(factor[:-1]+factor[1:])/2)
    cruise=float(np.sum(np.diff(distance)/segment_factor)/((n-1)/fps))
    time=np.r_[0,np.cumsum(np.diff(distance)/(cruise*segment_factor))]
    travelled=np.interp(np.linspace(0,time[-1],n),time,distance)
    xy=np.column_stack([np.interp(travelled,distance,smooth[:,j]) for j in range(2)])
    step=np.diff(xy,axis=0)
    speed=float(np.linalg.norm(step,axis=1).max()*fps)
    heading=np.unwrap(np.arctan2(step[:,1],step[:,0]))
    turn_rate=float(np.rad2deg(np.abs(np.diff(heading))).max()*fps) if n>2 else 0.
    feasible=speed<=float(plan['max_speed_mps']) and turn_rate<=float(plan.get('max_turn_rate_deg_s',35.))
    result={'xy':xy,'smooth':smooth,'smoothing_m':smooth_m,'drawn_length_m':drawn_length,
            'optimized_distance_m':float(distance[-1]),'max_speed_mps':speed,
            'max_turn_rate_deg_s':turn_rate,'turn_slowdown_distance_m':slow_distance,
            'turn_slowdown_scale':1.,'slowest_turn_speed_mps':float(cruise*segment_factor.min()),
            'straight_speed_mps':cruise,'feasible':feasible}
    if not feasible:
        result['error']=f'Drawn route exceeds limits: {speed:.1f} m/s, turn {turn_rate:.1f} deg/s'
    return result


def make_manual_route(cfg, terrain, bounds):
    plan=cfg['trajectory']
    route_path=Path(plan['manual_route_file'])
    if not route_path.is_file():
        raise FileNotFoundError(f'Draw and save a route first: {route_path}')
    route=json.loads(route_path.read_text())
    points=np.asarray(route['points_world_xy_m'],float)
    bounds=np.asarray(bounds,float)
    drawn_bounds=np.asarray(route['dop_bounds_local'],float)+np.tile(np.asarray(route['utm_offset'][:2],float),2)
    if not np.allclose(drawn_bounds,bounds,atol=.01):
        raise ValueError(f'Drawn route belongs to a different DOP crop: {route_path}')
    frames=int(cfg['sequence']['frames'])
    result=preview_route(points,bounds,frames,cfg['sequence']['fps'],plan)
    if not result['feasible']:
        raise ValueError(result['error'])
    xy=result['xy']
    if np.any(xy<bounds[:2]) or np.any(xy>bounds[2:]):
        raise ValueError('Smoothed route leaves its DOP')
    plan['route_optimization']={k:v for k,v in result.items() if k not in ('xy','smooth','feasible','score','error')}
    return np.column_stack((xy,np.full(len(xy),terrain+float(plan['height_agl_m']))))


def challenge_report(cfg, positions, rotations, agl):
    steps=np.linalg.norm(np.diff(positions,axis=0),axis=1)
    tilt=np.rad2deg(np.arccos(np.clip(-rotations[:,2,2],-1,1)))
    return {'route_source':'manual_2d','frames':len(positions),'distance_m':float(steps.sum()),
            'route_optimization':cfg['trajectory']['route_optimization'],
            'target_agl_m':cfg['trajectory']['height_agl_m'],
            'reference_agl_min_m':cfg['trajectory']['reference_agl_min_m'],
            'reference_agl_m':cfg['trajectory']['reference_agl_m'],
            'reference_agl_max_m':cfg['trajectory']['reference_agl_max_m'],
            'reference_tilt_from_nadir_deg':cfg['trajectory']['tilt_from_nadir_deg'],
            'camera_tilt_lock_deg':cfg['trajectory']['camera_tilt_lock_deg'],
            'agl_min_m':float(agl.min()),'agl_max_m':float(agl.max()),
            'tilt_min_deg':float(tilt.min()),'tilt_median_deg':float(np.median(tilt)),
            'tilt_max_deg':float(tilt.max())}
