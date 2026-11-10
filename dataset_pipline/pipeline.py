"""MovingDrone long-sequence generator. Run from the OrthoTrack repository root."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset_pipline.trajectory import make_forward_poses,export_ges,plan_altitude
from dataset_pipline.planner import make_manual_route,challenge_report
from dataset_pipline.collision import dsm_heights,validate as validate_collision
from dataset_pipline.loader import SceneLoader,create_output_dirs
from dataset_pipline.utils import (atomic_json,compact_output,check_video,check_depth,
                                   crop_dop,frustum_hits)
from utils.pose import PoseLoader


def prepare_manual(cfg, scene, assets):
    ref=assets.root;meta=assets.meta
    tilt=float(cfg['trajectory']['tilt_from_nadir_deg'])
    minimum_tilt=float(cfg['trajectory'].get('minimum_tilt_from_nadir_deg',20.))
    maximum_tilt=float(cfg['trajectory'].get('maximum_tilt_from_nadir_deg',30.))
    if tilt<minimum_tilt:
        raise ValueError(f'Camera tilt {tilt:.1f}° is below the {minimum_tilt:.1f}° minimum; '
                         'this would look too close to a satellite view. Move the route/crop inward '
                         'or expand map coverage before rendering a more forward-facing view.')
    if tilt>maximum_tilt:
        raise ValueError(f'Camera tilt {tilt:.1f}° exceeds the {maximum_tilt:.1f}° maximum.')
    frames=cfg['sequence']['frames']
    offset=np.asarray(meta['utm_offset'],float)
    dsm,dsm_bounds=assets.dsm();dsm_bounds=np.asarray(dsm_bounds,float)
    dop_info,image=assets.dop(cfg['map']['dop_year'])
    db=np.asarray(dop_info['bounds'],float)
    # The editor has already cropped the DOP. Its image and bounds are the
    # exact map used for drawing, so planning must use them without recropping.
    crop_bounds=db
    valid=np.isfinite(dsm)&(dsm>-9000)
    world_bounds=crop_bounds+np.tile(offset[:2],2)
    terrain=float(np.nanmedian(dsm[valid]))+offset[2]+cfg['map']['vertical_offset_m']
    look_center=(world_bounds[:2]+world_bounds[2:])/2
    look_center+=np.asarray(cfg['trajectory'].get('look_at_center_offset_xy_m',[0.,0.]))
    cfg['trajectory'].update(look_at_z=terrain,
                             look_at_center_xy=look_center.tolist())
    positions=make_manual_route(cfg,terrain,world_bounds)
    frames=cfg['sequence']['frames']
    local=positions-offset
    roof=dsm_heights(dsm,dsm_bounds,local[:,:2])
    base=float(cfg['trajectory']['height_agl_m'])
    positions[:,2]=plan_altitude(roof,base+cfg['map']['vertical_offset_m'],
        cfg['trajectory']['safe_distance_m'],cfg['trajectory'].get('max_vertical_speed_mps',4.),
        cfg['sequence']['fps'])+offset[2]
    local=positions-offset
    roof=dsm_heights(dsm,dsm_bounds,local[:,:2]);agl=local[:,2]-roof
    cfg['trajectory']['look_at_z']=float(np.nanmedian(roof))+offset[2]+cfg['map']['vertical_offset_m']
    mesh_vertices,mesh_faces=assets.mesh()
    collision=validate_collision(local,dsm,dsm_bounds,mesh_vertices,mesh_faces,
                                 cfg['trajectory']['safe_distance_m'])
    speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)*cfg['sequence']['fps']
    if speed.max()>cfg['trajectory']['max_speed_mps']*1.01:
        i=int(np.argmax(speed));h=float(np.linalg.norm(np.diff(positions[:,:2],axis=0)[i])*cfg['sequence']['fps'])
        v=float(abs(np.diff(positions[:,2])[i])*cfg['sequence']['fps'])
        raise ValueError(f'Manual route exceeds max speed at frame {i}: {speed[i]:.2f} m/s (horizontal {h:.2f}, vertical {v:.2f})')
    floor=float(np.nanmedian(roof))
    mesh_xy=mesh_vertices[:,:2];mesh_bounds=np.r_[mesh_xy.min(0),mesh_xy.max(0)]
    texture_bounds=np.asarray(meta.get('render_dop',dop_info)['bounds'],float)
    cfg['trajectory']['camera_tilt_lock_deg']=float(cfg['trajectory']['tilt_from_nadir_deg'])
    rot=make_forward_poses(positions,cfg)
    hits=frustum_hits(local,rot,floor,cfg,dsm,dsm_bounds)
    ground_hits=hits[np.isfinite(hits[:,:,0])]
    hit_bounds=np.r_[ground_hits.min(0),ground_hits.max(0)]
    render_bounds=np.r_[np.minimum(crop_bounds[:2],hit_bounds[:2]),
                        np.maximum(crop_bounds[2:],hit_bounds[2:])]
    if (np.any(render_bounds[:2]<dsm_bounds[:2]+2.) or np.any(render_bounds[2:]>dsm_bounds[2:]-2.) or
        np.any(render_bounds[:2]<mesh_bounds[:2]+2.) or np.any(render_bounds[2:]>mesh_bounds[2:]-2.) or
        np.any(render_bounds[:2]<texture_bounds[:2]+2.) or np.any(render_bounds[2:]>texture_bounds[2:]-2.)):
        raise ValueError(f'View bounds {render_bounds.round(1).tolist()} exceed DSM {dsm_bounds.round(1).tolist()}, '
                         f'mesh {mesh_bounds.round(1).tolist()}, or DOP {texture_bounds.round(1).tolist()}')
    inside=((hits[:,:,0]>=crop_bounds[0])&(hits[:,:,0]<=crop_bounds[2])&
            (hits[:,:,1]>=crop_bounds[1])&(hits[:,:,1]<=crop_bounds[3]))
    overlap=float(inside[np.isfinite(hits[:,:,0])].mean())
    if overlap<cfg['trajectory'].get('min_view_overlap_fraction',.35):
        raise ValueError(f'Camera view overlap with selected tile is {overlap:.1%}')
    view_points=np.concatenate((ground_hits,local[:,:2]),axis=0)
    view_bounds=np.r_[view_points.min(0)-10.,view_points.max(0)+10.]
    center=(crop_bounds[:2]+crop_bounds[2:])/2
    half=(crop_bounds[2:]-crop_bounds[:2])*cfg['trajectory'].get('max_view_crop_scale',2.)/2
    view_bounds[:2]=np.maximum(np.maximum(view_bounds[:2],center-half),texture_bounds[:2])
    view_bounds[2:]=np.minimum(np.minimum(view_bounds[2:],center+half),texture_bounds[2:])
    cfg['sequence']['name']=scene.parent.name+'_'+scene.name
    doc=export_ges(scene/'trajectory.json',positions,rot,cfg)
    loader=PoseLoader(str(scene/'trajectory.json'),utm_zone=cfg['map']['utm_zone'])
    full_dop,full_image=assets.dop(cfg['map']['dop_year'],render=True)
    view_crop,view_bounds=crop_dop(full_image,np.asarray(full_dop['bounds'],float),view_bounds)
    view_path=scene/'dop_view.jpg';tmp_view=scene/'dop_view.tmp.jpg'
    if not cv2.imwrite(str(tmp_view),view_crop): raise IOError(f'Cannot save UAV-view DOP crop: {tmp_view}')
    tmp_view.replace(view_path)
    dop_dir=scene/'dop'
    dop_link=dop_dir/f'{cfg["map"]["dop_year"]}.jpg'
    dop_link.unlink(missing_ok=True)
    dop_link.symlink_to(view_path.resolve())
    dsm_link=scene/'dsm.npz'
    dsm_link.unlink(missing_ok=True)
    dsm_link.symlink_to((ref/'dsm.npz').resolve())
    vf=np.deg2rad(cfg['sequence']['fov_vertical_deg'])
    fy=cfg['sequence']['height']/(2*np.tan(vf/2))
    atomic_json(scene/'intrinsics.json',{'width':cfg['sequence']['width'],'height':cfg['sequence']['height'],
                                        'fx':fy,'fy':fy,'cx':cfg['sequence']['width']/2,'cy':cfg['sequence']['height']/2})
    output_meta={'name':cfg['sequence']['name'],'width':cfg['sequence']['width'],
        'height':cfg['sequence']['height'],'num_frames':frames,'frame_rate':cfg['sequence']['fps'],
        'utm_zone':cfg['map']['utm_zone'],'utm_offset':offset.tolist(),'dsm':meta['dsm'],
        'dops':{f'dop_{cfg["map"]["dop_year"]}':{'year':cfg['map']['dop_year'],
        'file':f'dop/{cfg["map"]["dop_year"]}.jpg','bounds':view_bounds.tolist(),
        'gsd':float((full_dop['bounds'][2]-full_dop['bounds'][0])/full_image.shape[1])}},
        'selected_dop':{'file':'dop.jpg','bounds':crop_bounds.tolist()},
        'uav_view_bounds_local':view_bounds.tolist(),'render_bounds_local':render_bounds.tolist(),
        'vertical_offset_m':cfg['map']['vertical_offset_m']}
    if (scene/'video.mp4').is_file(): output_meta['renderer']='open3d_gpu'
    depth_meta=scene/'depth'/'metadata.json'
    if depth_meta.is_file():
        output_meta['depth']={'directory':'depth','file_pattern':'depth_%06d.png',**json.loads(depth_meta.read_text())}
    atomic_json(scene/'meta.json',output_meta)
    atomic_json(scene/'config_used.json',cfg)
    report=challenge_report(cfg,local,rot,agl)
    report['collision']=collision
    report['render_bounds_local']=render_bounds.tolist();report['crop_bounds_local']=crop_bounds.tolist()
    report['uav_view_crop_bounds_local']=view_bounds.tolist()
    report['fov_vertical_deg']=float(cfg['sequence']['fov_vertical_deg'])
    report['camera_tilt_lock_deg']=float(cfg['trajectory']['camera_tilt_lock_deg'])
    report['frustum_ray_grid']=5
    report['frustum_inside_dop_fraction']=overlap
    view_inside=((hits[:,:,0]>=view_bounds[0])&(hits[:,:,0]<=view_bounds[2])&
                 (hits[:,:,1]>=view_bounds[1])&(hits[:,:,1]<=view_bounds[3]))
    report['frustum_inside_view_crop_fraction']=float(view_inside[np.isfinite(hits[:,:,0])].mean())
    atomic_json(scene/'plan_report.json',report)
    preview=scene/'preview'
    fig,ax=plt.subplots(figsize=(7,7))
    ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),extent=crop_bounds[[0,2,1,3]],origin='upper')
    ax.plot(local[:,0],local[:,1],color='#ffeb3b',linewidth=2.4,
            path_effects=[path_effects.Stroke(linewidth=4,foreground='#17202a'),path_effects.Normal()])
    fig.savefig(preview/'trajectory_on_dop.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,7))
    ax.imshow(cv2.cvtColor(view_crop,cv2.COLOR_BGR2RGB),extent=view_bounds[[0,2,1,3]],origin='upper')
    ax.plot(local[:,0],local[:,1],color='#ffeb3b',linewidth=2.4,
            path_effects=[path_effects.Stroke(linewidth=4,foreground='#17202a'),path_effects.Normal()])
    fig.savefig(preview/'trajectory_on_view_dop.png',dpi=150);plt.close(fig)
    with (scene/'poses.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['frame_id','x','y','z','qw','qx','qy','qz','latitude','longitude','altitude','fov_vertical'])
        for i,p in enumerate(loader.poses):
            c=doc['cameraFrames'][i]['coordinate']
            writer.writerow([i,p.x-offset[0],p.y-offset[1],p.z-offset[2],p.qw,p.qx,p.qy,p.qz,
                             c['latitude'],c['longitude'],c['altitude']-offset[2],doc['cameraFrames'][i]['fovVertical']])
    return doc


def render_first_frame_gpu(cfg,scene,doc,assets):
    import open3d as o3d
    from open3d.visualization import rendering
    offset=np.asarray(assets.meta['utm_offset'],np.float32)
    vertices,faces=assets.mesh();dop,image=assets.dop(cfg['map']['dop_year'],render=True)
    bounds=np.asarray(dop['bounds'],float)
    mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
    tri_xy=vertices[faces.reshape(-1),:2]
    mesh.triangle_uvs=o3d.utility.Vector2dVector(np.column_stack(((tri_xy[:,0]-bounds[0])/(bounds[2]-bounds[0]),
                                                                  (tri_xy[:,1]-bounds[1])/(bounds[3]-bounds[1]))))
    material=rendering.MaterialRecord();material.shader='defaultUnlit'
    material.albedo_img=o3d.geometry.Image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    renderer=rendering.OffscreenRenderer(cfg['sequence']['width'],cfg['sequence']['height'])
    renderer.scene.set_background([0.53,0.78,0.92,1.]);renderer.scene.add_geometry('city',mesh,material)
    renderer.scene.camera.set_projection(cfg['sequence']['fov_vertical_deg'],
        cfg['sequence']['width']/float(cfg['sequence']['height']),.1,100000.,rendering.Camera.FovType.Vertical)
    pose=PoseLoader(str(scene/'trajectory.json'),utm_zone=cfg['map']['utm_zone']).get_pose(0)
    position=(pose.position-offset).astype(np.float32);rotation=pose.rotation_matrix
    forward=rotation@np.array([0.,0.,1.]);up=rotation@np.array([0.,-1.,0.])
    up-=np.dot(up,forward)*forward;up/=np.linalg.norm(up)
    renderer.scene.camera.look_at(position+forward*10000.,position,up)
    frame=np.asarray(renderer.render_to_image())
    path=scene/'preview'/'first_frame_rgb.png'
    if not cv2.imwrite(str(path),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)):
        raise IOError(f'Cannot save first-frame preview: {path}')
    return path


def render_gpu(cfg,scene,doc,assets,save_depth=False,save_rgb=True):
    """Render the DOP-textured mesh and optional depth with Open3D's GPU renderer."""
    import open3d as o3d
    from open3d.visualization import rendering
    meta=assets.meta
    offset=np.asarray(meta['utm_offset'],np.float32)
    vertices,faces=assets.mesh()
    dop,image=assets.dop(cfg['map']['dop_year'],render=True)
    bounds=np.asarray(dop['bounds'],float);h=cfg['sequence']['height'];w=cfg['sequence']['width']
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
    tri_xy=vertices[faces.reshape(-1),:2]
    mesh.triangle_uvs=o3d.utility.Vector2dVector(np.column_stack((
        (tri_xy[:,0]-bounds[0])/(bounds[2]-bounds[0]),
        (tri_xy[:,1]-bounds[1])/(bounds[3]-bounds[1]))))
    material=rendering.MaterialRecord();material.shader='defaultUnlit'
    material.albedo_img=o3d.geometry.Image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    renderer=rendering.OffscreenRenderer(w,h)
    renderer.scene.set_background([0.53,0.78,0.92,1.])
    renderer.scene.add_geometry('city',mesh,material)
    renderer.scene.camera.set_projection(cfg['sequence']['fov_vertical_deg'],w/float(h),.1,100000.,
                                         rendering.Camera.FovType.Vertical)
    loader=PoseLoader(str(scene/'trajectory.json'),utm_zone=cfg['map']['utm_zone'])
    n=len(doc['cameraFrames']);fps=cfg['sequence']['fps'];depth_dir=scene/'depth'
    video=scene/'video.mp4';partial=scene/'video.partial.mp4'
    writer=cv2.VideoWriter(str(partial),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)) if save_rgb else None
    if writer is not None and not writer.isOpened():raise RuntimeError('Cannot open video writer')
    samples=[];sample_ids=set(np.linspace(0,n-1,min(9,n),dtype=int))
    try:
        for i in range(n):
            pose=loader.get_pose(i);rotation=pose.rotation_matrix
            position=(pose.position-offset).astype(np.float32)
            forward=rotation@np.array([0.,0.,1.]);up=rotation@np.array([0.,-1.,0.])
            up-=np.dot(up,forward)*forward;up/=np.linalg.norm(up)
            renderer.scene.camera.look_at(position+forward*10000.,position,up)
            if save_rgb:
                frame=np.asarray(renderer.render_to_image())
                writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            if save_depth:
                depth=np.asarray(renderer.render_to_depth_image(z_in_view_space=True),np.float32)
                valid=np.isfinite(depth)&(depth>0)&(depth<99999.)
                scale=float(cfg.get('depth',{}).get('scale_m_per_unit',.01))
                if np.any(depth[valid]>65535*scale):
                    raise ValueError(f'Frame {i} depth exceeds {65535*scale:.2f} m')
                encoded=np.zeros((h,w),np.uint16)
                encoded[valid]=np.maximum(1,np.rint(depth[valid]/scale)).astype(np.uint16)
                target=depth_dir/f'depth_{i:06d}.png';tmp=depth_dir/f'depth_{i:06d}.tmp.png'
                if not cv2.imwrite(str(tmp),encoded,[cv2.IMWRITE_PNG_COMPRESSION,3]):
                    raise IOError(f'Failed to save {tmp}')
                tmp.replace(target)
            if save_rgb and i in sample_ids:samples.append(cv2.resize(frame,(320,180)))
            if (i+1)%200==0 or i+1==n:print(f'GPU render: {i+1}/{n}',flush=True)
        if writer is not None:
            writer.release();check_video(partial,n,cfg)
    except Exception:
        if writer is not None:writer.release();partial.unlink(missing_ok=True)
        raise
    if save_rgb:partial.replace(video)
    meta_path=scene/'meta.json';output_meta=json.loads(meta_path.read_text())
    if save_rgb:output_meta['renderer']='open3d_gpu'
    if save_depth:
        scale=float(cfg.get('depth',{}).get('scale_m_per_unit',.01))
        depth_meta={'frames':n,'width':w,'height':h,'format':'PNG','dtype':'uint16',
                    'depth_type':'camera_z','unit':'m','scale_m_per_unit':scale,
                    'invalid_value':0,'valid_range_m':[scale,65535*scale]}
        atomic_json(depth_dir/'metadata.json',depth_meta);check_depth(scene,n,cfg)
        output_meta['depth']={'directory':'depth','file_pattern':'depth_%06d.png',**depth_meta}
    atomic_json(meta_path,output_meta)
    if samples:
        rows=[np.hstack(samples[j:j+3]) for j in range(0,len(samples),3)]
        cv2.imwrite(str(scene/'preview'/'camera_view_samples.jpg'),np.vstack(rows))
    compact_output(scene)


def load_config(path, frames=None):
    cfg=yaml.safe_load(path.read_text())
    route=Path(cfg['trajectory']['manual_route_file'])
    cfg['trajectory']['manual_route_sha256']=hashlib.sha256(route.read_bytes()).hexdigest()
    cfg['sequence']['frames']=int(frames or cfg['sequence']['frames'])
    return cfg


def run(cfg, command, save_depth=False, save_rgb=True):
    frames=cfg['sequence']['frames'];scene=Path(cfg['map']['tile_dir'])
    if command=='validate':
        cfg=json.loads((scene/'config_used.json').read_text())
        frames=cfg['sequence']['frames']
        check_video(scene/'video.mp4',frames,cfg)
        meta=json.loads((scene/'meta.json').read_text())
        if meta.get('renderer')!='open3d_gpu': raise ValueError('Video was not rendered with the GPU pipeline')
        if save_depth: check_depth(scene,frames,cfg)
        with (scene/'poses.csv').open() as f: rows=list(csv.DictReader(f))
        if len(rows)!=frames: raise ValueError('Pose row count mismatch')
        positions=np.asarray([[float(row[k]) for k in ('x','y','z')] for row in rows])
        if not np.isfinite(positions).all(): raise ValueError('Non-finite pose position')
        speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)*cfg['sequence']['fps']
        if len(speed) and speed.max()>cfg['trajectory']['max_speed_mps']*1.01:
            raise ValueError(f'Speed discontinuity: {speed.max():.2f} m/s')
        print(f'Validated {frames} manual-route frames: {scene}')
        return
    assets=SceneLoader(cfg['map']['reference_scene'])
    create_output_dirs(scene,depth=save_depth and command=='generate',
                       clear_depth=save_depth and command=='generate')
    doc=prepare_manual(cfg,scene,assets)
    frames=cfg['sequence']['frames']
    if command=='plan':
        print(f'Planned {frames} manual-route frames: {scene}')
        return
    if command=='preview':
        path=render_first_frame_gpu(cfg,scene,doc,assets)
        print(f'First-frame RGB preview: {path}')
        return
    render_gpu(cfg,scene,doc,assets,save_depth,save_rgb)
    print(f'Complete: {scene}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['plan','preview','generate','validate','plan-all','generate-all','validate-all'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--root',type=Path,default=Path('/mnt/Data/RIght/MovingDrone_Tiles'))
    parser.add_argument('--frames',type=int)
    parser.add_argument('--depth',action='store_true',help='Save 16-bit metric camera-Z depth PNGs')
    parser.add_argument('--no-rgb',dest='save_rgb',action='store_false',default=True,
                        help='Skip RGB video output and render only auxiliary outputs such as depth')
    args=parser.parse_args()
    if args.command.startswith('generate') and not args.depth and not args.save_rgb:
        parser.error('Select RGB and/or --depth')
    if args.command.endswith('-all'):
        action=args.command[:-4]
        configs=[]
        for path in sorted(args.root.glob('*/*/config.yaml')):
            route=yaml.safe_load(path.read_text())['trajectory'].get('manual_route_file')
            if route and Path(route).is_file(): configs.append(path)
        if not configs: raise FileNotFoundError(f'No tiles with saved hand-drawn routes under {args.root}')
        failures=[]
        for i,path in enumerate(configs,1):
            try:
                cfg=load_config(path,args.frames)
                print(f'[{i}/{len(configs)}] {path.parent.parent.name}/{path.parent.name}',flush=True)
                run(cfg,action,args.depth,args.save_rgb)
            except Exception as error:
                failures.append((path,str(error)))
                print(f'FAILED {path}: {error}',flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)}/{len(configs)} tiles failed: '+', '.join(p.parent.name for p,_ in failures))
        return
    if args.config is None: parser.error('--config is required unless using an *-all command')
    run(load_config(args.config,args.frames),args.command,args.depth,args.save_rgb)


if __name__=='__main__':
    main()
