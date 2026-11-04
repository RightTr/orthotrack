"""Procedural metric routes; no hand-authored camera waypoints or Google Earth input."""
import numpy as np


def make_route(cfg, center, terrain, bounds):
    return make_forward_sweep(cfg, center, terrain, bounds)


def make_forward_sweep(cfg, center, terrain, bounds):
    """Continuous lawnmower route, sampled by travel time rather than frame count."""
    plan = cfg['trajectory']; n = cfg['sequence']['frames']; fps = cfg['sequence']['fps']
    if n < 2 or fps <= 0: raise ValueError('Forward sweep requires at least two frames and positive FPS')
    cruise = float(plan['cruise_speed_mps']); turn = float(plan['turn_speed_mps'])
    if not 0 < turn <= cruise <= plan['max_speed_mps']: raise ValueError('Invalid cruise/turn/max speeds')
    b = np.asarray(bounds, float)
    margin = float(plan['edge_margin_ratio']); coverage = float(plan['coverage_ratio'])
    if not 0 < coverage <= 1-2*margin or not 0 <= margin < .5:
        raise ValueError('Coverage must fit inside the edge margin')
    size = b[2:]-b[:2]; swap = size[1] > size[0]
    long, short = (size[::-1] if swap else size)
    long_span = coverage*long; short_span = coverage*short
    # Enough distinct lanes to cover the short axis at least once per sweep.
    lanes = max(3, int(np.ceil((n-1)/fps*cruise/(2*long_span))))
    spacing = short_span/(lanes-1)
    radius = spacing*.65  # Includes the largest adjacent-lane jitter difference.
    if long_span <= 2*radius: raise ValueError('Crop too narrow for smooth sweep turns')
    xend = long_span/2-radius
    low, high = -short_span/2, short_span/2
    rng = np.random.default_rng(plan['seed'])
    # Small deterministic lane offsets make the return sweep visit new ground.
    jitter = rng.uniform(-.12,.12,lanes)*spacing
    jitter[0]=jitter[-1]=0
    ys = np.linspace(low,high,lanes)+jitter
    points=[]; speeds=[]; duration=(n-1)/fps
    direction=1; index=0; elapsed=0.; passes=0
    def append(segment, speed):
        nonlocal elapsed
        if points: segment=segment[1:]
        if len(segment)==0: return
        if points:
            elapsed += np.linalg.norm(segment[0]-points[-1])/speed
        if len(segment)>1:
            elapsed += np.linalg.norm(np.diff(segment,axis=0),axis=1).sum()/speed
        points.extend(segment); speeds.extend([speed]*len(segment))
    while elapsed < duration+2*long_span/turn:
        y=ys[index]
        x0=-direction*xend; x1=direction*xend
        append(np.column_stack((np.linspace(x0,x1,max(3,int(2*xend/2)+1)),
                                np.full(max(3,int(2*xend/2)+1),y))),cruise)
        next_index=index+1 if passes//(lanes-1)%2==0 else index-1
        if next_index < 0 or next_index >= lanes:
            next_index=index-1 if index==lanes-1 else index+1
        next_y=ys[next_index]; r=abs(next_y-y)/2; mid=(y+next_y)/2
        theta=np.linspace(0,np.pi,max(12,int(np.pi*r/1.5)+1))
        arc=np.column_stack((direction*(xend+r*np.sin(theta)),
                             mid-np.sign(next_y-y)*r*np.cos(theta)))
        append(arc,turn)
        direction*=-1; index=next_index; passes+=1
    xy=np.asarray(points); v=np.asarray(speeds)
    steps=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    times=np.r_[0,np.cumsum(steps/np.minimum(v[:-1],v[1:]))]
    sample=np.arange(n)/fps
    xy=np.column_stack([np.interp(sample,times,xy[:,j]) for j in range(2)])
    if swap: xy=xy[:,::-1]
    xy += (b[:2]+b[2:])/2
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    amp=float(plan['height_variation_m']); wavelength=float(plan['height_wavelength_m'])
    if amp < 0 or wavelength <= 0: raise ValueError('Invalid height amplitude or wavelength')
    phase=rng.uniform(-np.pi,np.pi)
    z=terrain+float(plan['height_agl_m'])+amp*np.sin(2*np.pi*distance/wavelength+phase)
    plan['effective_passes']=passes
    return np.column_stack((xy,z))


def challenge_report(cfg, positions, rotations, agl, coverage_scale):
    n = len(positions)
    steps = np.linalg.norm(np.diff(positions,axis=0),axis=1)
    heading = np.unwrap(np.arctan2(rotations[:,1,2],rotations[:,0,2]))
    turn = np.abs(np.diff(heading))
    tilt = np.rad2deg(np.arccos(np.clip(-rotations[:,2,2],-1,1)))
    passes=int(cfg['trajectory'].get('effective_passes',1))
    edges = np.linspace(0,n,passes+1,dtype=int)
    segments = []
    for i,(start,end) in enumerate(zip(edges[:-1],edges[1:])):
        segments.append({'segment':i+1,'start_frame':int(start),'end_frame':int(end),
                         'distance_m':float(steps[start:max(start,end-1)].sum()),
                         'agl_min_m':float(agl[start:end].min()),
                         'agl_max_m':float(agl[start:end].max()),
                         'heading_change_deg':float(np.rad2deg(turn[start:max(start,end-1)].sum()))})
    return {'seed':cfg['trajectory']['seed'],'passes':passes,'frames':n,'coverage_scale':coverage_scale,
            'reference_agl_m':cfg['trajectory'].get('reference_agl_m'),
            'planned_height_base_m':cfg['trajectory']['height_agl_m'],
            'planned_height_amplitude_m':cfg['trajectory']['height_variation_m'],
            'distance_m':float(steps.sum()),'agl_min_m':float(agl.min()),
            'agl_max_m':float(agl.max()),'scale_ratio':float(agl.max()/agl.min()),
            'tilt_min_deg':float(tilt.min()),'tilt_max_deg':float(tilt.max()),
            'max_turn_rate_deg_s':float(np.rad2deg(turn.max())*cfg['sequence']['fps']),
            'segments':segments}
