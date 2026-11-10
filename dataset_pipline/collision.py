"""Conservative DSM and render-mesh clearance checks in one local metric frame."""
import numpy as np


def dsm_heights(dsm, bounds, xy):
    h,w =dsm.shape; x0,y0,x1,y1=bounds
    col=(xy[:,0]-x0)/(x1-x0)*(w-1); row=(y1-xy[:,1])/(y1-y0)*(h-1)
    if np.any((col<0)|(col>=w-1)|(row<0)|(row>=h-1)):
        raise ValueError('Trajectory exits DSM coverage')
    c=col.astype(int); r=row.astype(int); dx=col-c; dy=row-r
    return (1-dx)*(1-dy)*dsm[r,c]+dx*(1-dy)*dsm[r,c+1]+(1-dx)*dy*dsm[r+1,c]+dx*dy*dsm[r+1,c+1]


def validate(points, dsm, bounds, vertices, faces, clearance=3., step=.5):
    """Returns minimum clearance; fails closed on missing/invalid geometry."""
    pieces=[]; owners=[]
    for i in range(len(points)-1):
        count=max(2,int(np.ceil(np.linalg.norm(points[i+1]-points[i])/step))+1)
        pieces.append(points[i]+np.linspace(0,1,count,endpoint=False)[:,None]*(points[i+1]-points[i]))
        owners.extend([i]*count)
    samples=np.concatenate(pieces+[points[-1:]])
    owners=np.asarray(owners+[len(points)-1])
    surface=dsm_heights(dsm,bounds,samples[:,:2]); valid=np.isfinite(surface)&(surface>-9000)
    if not valid.all(): raise ValueError(f'DSM void near frame {int(owners[~valid][0])}')
    gap=samples[:,2]-surface
    if gap.min()<=clearance: raise ValueError(f'DSM clearance at frame {int(owners[gap.argmin()])}: {gap.min():.2f} m')
    mesh_top=float(vertices[:,2].max())
    if points[:,2].min()>mesh_top+clearance:
        return {'min_clearance_m':float(min(gap.min(),points[:,2].min()-mesh_top)),
                'collision_count':0,'invalid_frames':[],'adjusted_segments':[]}
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(vertices),o3d.core.Tensor(faces))
    minimum=float('inf')
    for start in range(0,len(samples),100000):
        query=o3d.core.Tensor(samples[start:start+100000].astype('float32'))
        distances=scene.compute_distance(query).numpy(); minimum=min(minimum,float(distances.min()))
        if (distances<=clearance+step/2).any():
            j=np.flatnonzero(distances<=clearance+step/2)[0]+start
            raise ValueError(f'Mesh clearance at frame {int(owners[j])}: {distances[j-start]:.2f} m')
    return {'min_clearance_m':min(minimum,float(gap.min())),'collision_count':0,'invalid_frames':[],'adjusted_segments':[]}
