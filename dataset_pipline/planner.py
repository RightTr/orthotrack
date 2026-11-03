"""Procedural metric routes; no hand-authored camera waypoints or Google Earth input."""
import numpy as np


def make_route(cfg, center, terrain, bounds):
    n = cfg['sequence']['frames']
    plan = cfg['trajectory']
    rng = np.random.default_rng(plan['seed'])
    passes = max(2,int(plan.get('passes_override',plan['passes'])))
    plan['effective_passes']=passes
    if passes < 2 or n < 2:
        raise ValueError('Auto planner needs at least two passes and two frames')
    width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
    x_span = plan['route_span_x_ratio']*width*rng.uniform(.9,1.1)
    y_span = plan['route_span_y_ratio']*height*rng.uniform(.9,1.1)
    phase_x,phase_y,phase_z=rng.uniform(-np.pi,np.pi,3)
    curve_a,curve_b,cross_a=rng.uniform(.28,.42),rng.uniform(.1,.2),rng.uniform(.7,.9)
    dense_u = np.linspace(0,1,max(50000,5*n))
    # Alternating long traverses; the Y program selects the trajectory ablation.
    x = x_span*np.cos(passes*np.pi*dense_u+phase_x)
    shape = plan['route_shape']
    if shape == 'straight':
        y = y_span*(2*dense_u-1)
    elif shape == 'curved':
        y = (y_span*(2*dense_u-1)
             + curve_a*y_span*np.sin(2*passes*np.pi*dense_u+phase_y)
             + curve_b*y_span*np.sin(4*np.pi*dense_u+phase_y/2))
    elif shape == 'crossing':
        y = (.35*y_span*(2*dense_u-1)
             + cross_a*y_span*np.sin(3*np.pi*dense_u+phase_y))
    else:
        raise ValueError(f'Unknown route_shape: {shape}')
    dense = np.column_stack((x,y))
    arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    target = np.linspace(0,arc[-1],n)
    xy = np.column_stack([np.interp(target,arc,dense[:,axis]) for axis in range(2)])
    # Reserve speed for climbs and descents; keep the total travel time fixed.
    duration = (n-1)/cfg['sequence']['fps']
    max_horizontal = .95*plan['max_speed_mps']*duration
    xy *= min(1.,max_horizontal/arc[-1])
    u = target/arc[-1]
    z = (terrain + plan['height_agl_m']
         - plan['height_variation_m']*np.cos(4*np.pi*u+phase_z)
         + .2*plan['height_variation_m']*np.sin(10*np.pi*u+phase_y))
    return np.column_stack((center[0]+xy[:,0],center[1]+xy[:,1],z))


def challenge_report(cfg, positions, rotations, agl, coverage_scale):
    n = len(positions)
    steps = np.linalg.norm(np.diff(positions,axis=0),axis=1)
    heading = np.unwrap(np.arctan2(rotations[:,1,2],rotations[:,0,2]))
    turn = np.abs(np.diff(heading))
    tilt = np.rad2deg(np.arccos(np.clip(-rotations[:,2,2],-1,1)))
    passes=int(cfg['trajectory'].get('effective_passes',cfg['trajectory']['passes']))
    edges = np.linspace(0,n,passes+1,dtype=int)
    segments = []
    for i,(start,end) in enumerate(zip(edges[:-1],edges[1:])):
        segments.append({'segment':i+1,'start_frame':int(start),'end_frame':int(end),
                         'distance_m':float(steps[start:max(start,end-1)].sum()),
                         'agl_min_m':float(agl[start:end].min()),
                         'agl_max_m':float(agl[start:end].max()),
                         'heading_change_deg':float(np.rad2deg(turn[start:max(start,end-1)].sum()))})
    return {'seed':cfg['trajectory']['seed'],'passes':passes,'frames':n,'coverage_scale':coverage_scale,
            'reference_agl_m':cfg['trajectory']['reference_agl_m'],
            'planned_height_base_m':cfg['trajectory']['height_agl_m'],
            'planned_height_amplitude_m':cfg['trajectory']['height_variation_m'],
            'distance_m':float(steps.sum()),'agl_min_m':float(agl.min()),
            'agl_max_m':float(agl.max()),'scale_ratio':float(agl.max()/agl.min()),
            'tilt_min_deg':float(tilt.min()),'tilt_max_deg':float(tilt.max()),
            'max_turn_rate_deg_s':float(np.rad2deg(turn.max())*cfg['sequence']['fps']),
            'segments':segments}
