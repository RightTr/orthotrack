"""Browser UI for drawing routes and generating saved DOP tiles."""
import argparse
import json
import subprocess
import sys
import traceback
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import unquote,urlparse

import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dataset_pipline.route_editor.tiles import accept_candidate,crop_candidate,scene_names,source_info
from dataset_pipline.loader import SceneLoader
from dataset_pipline.planner import preview_route
from dataset_pipline.trajectory import make_forward_poses,plan_altitude
from dataset_pipline.collision import dsm_heights
from dataset_pipline.utils import crop_dop,frustum_hits

PAGE=Path(__file__).with_name('index.html')
mesh_height_cache={}


def tile_catalog(root,ref_path):
    catalog={}
    pattern='tile*/config.yaml' if (root/'meta.json').is_file() or root.name in scene_names(ref_path) else '*/tile*/config.yaml'
    for path in sorted(root.glob(pattern)):
        cfg=yaml.safe_load(path.read_text())
        ref=Path(cfg['map']['reference_scene'])
        loader=SceneLoader(ref,load_dsm=False,load_mesh=False)
        meta=loader.meta
        dop=loader.dop_info()
        name=cfg['map']['scene_name']
        image=ref/dop['file']
        catalog[name]={'name':name,'scene':path.parent.parent.name,
                       'width':cfg['map']['crop_width_px'],'height':cfg['map']['crop_height_px'],
                       'bounds_local':dop['bounds'],'utm_offset':meta['utm_offset'],
                       'image':image,'route':Path(cfg['trajectory']['manual_route_file']),
                       'config':path,
                       'agl_min_m':cfg['trajectory']['reference_agl_min_m'],
                       'agl_m':cfg['trajectory']['reference_agl_m'],
                       'height_agl_m':cfg['trajectory']['height_agl_m'],
                       'agl_max_m':cfg['trajectory']['reference_agl_max_m'],
                       'tilt_deg':cfg['trajectory']['tilt_from_nadir_deg'],
                       'camera_center_weight':cfg['trajectory'].get('camera_center_weight',.5),
                       'heading_lookahead_m':cfg['trajectory'].get('heading_lookahead_m',35.),
                       'frames':cfg['sequence']['frames'],
                       'sequence':cfg['sequence'],'trajectory':cfg['trajectory']}
    return catalog


def dop_view_preview(tile,points,height,tilt,aim,frames):
    cfg=yaml.safe_load(tile['config'].read_text())
    cfg['sequence']['frames']=int(frames)
    plan=cfg['trajectory'];height=float(height);tilt=float(tilt)
    if not plan['reference_agl_min_m']<=height<=plan['reference_agl_max_m']:
        raise ValueError('Height must stay within the reference AGL range')
    if not 20<=tilt<=30:raise ValueError('Camera tilt must be between 20° and 30° from nadir')
    if not 0<=aim<=1:raise ValueError('Camera aim must be between path and tile center')
    plan['camera_center_weight']=aim
    x0,y0,x1,y1=tile['bounds_local']
    local_xy=np.column_stack((x0+points[:,0]*(x1-x0),y1-points[:,1]*(y1-y0)))
    offset=np.asarray(tile['utm_offset'],float);world_xy=local_xy+offset[:2]
    world_bounds=np.asarray(tile['bounds_local'],float)+np.tile(offset[:2],2)
    route=preview_route(world_xy,world_bounds,cfg['sequence']['frames'],cfg['sequence']['fps'],plan)
    loader=SceneLoader(cfg['map']['reference_scene'],load_mesh=False)
    dsm,dsm_bounds=loader.dsm();dsm_bounds=np.asarray(dsm_bounds,float)
    terrain=float(np.nanmedian(dsm[dsm>-9000]))+offset[2]+cfg['map']['vertical_offset_m']
    positions=np.column_stack((route['xy'],np.full(len(route['xy']),terrain+height)))
    local=positions-offset
    roof=dsm_heights(dsm,dsm_bounds,local[:,:2])
    local[:,2]=plan_altitude(roof,height+cfg['map']['vertical_offset_m'],
        plan['safe_distance_m'],plan['max_vertical_speed_mps'],cfg['sequence']['fps'])
    plan['look_at_center_xy']=((world_bounds[:2]+world_bounds[2:])/2-offset[:2]).tolist()
    plan['look_at_z']=float(np.nanmedian(roof)+cfg['map']['vertical_offset_m'])
    plan['camera_tilt_lock_deg']=tilt
    rotations=make_forward_poses(local,cfg)
    hits=frustum_hits(local,rotations,float(np.nanmedian(roof)),cfg,dsm,dsm_bounds)
    ground_hits=hits[np.isfinite(hits[:,:,0])]
    if not len(ground_hits):raise ValueError('Camera view does not intersect the ground')
    crop=np.asarray(tile['bounds_local'],float);center=(crop[:2]+crop[2:])/2
    half=(crop[2:]-crop[:2])*plan.get('max_view_crop_scale',5.)/2
    bounds=np.r_[np.minimum(ground_hits.min(0),local[:,:2].min(0))-10,
                 np.maximum(ground_hits.max(0),local[:,:2].max(0))+10]
    full_dop,full_image=loader.dop(cfg['map']['dop_year'],render=True)
    texture=np.asarray(full_dop['bounds'],float)
    bounds[:2]=np.maximum(np.maximum(bounds[:2],center-half),texture[:2])
    bounds[2:]=np.minimum(np.minimum(bounds[2:],center+half),texture[2:])
    view,actual=crop_dop(full_image,texture,bounds)
    for xy in route['xy']:
        px=int((xy[0]-actual[0])/(actual[2]-actual[0])*view.shape[1])
        py=int((actual[3]-xy[1])/(actual[3]-actual[1])*view.shape[0])
        if 0<=px<view.shape[1] and 0<=py<view.shape[0]:cv2.circle(view,(px,py),2,(0,255,255),-1)
    overlap=((hits[:,:,0]>=crop[0])&(hits[:,:,0]<=crop[2])&
             (hits[:,:,1]>=crop[1])&(hits[:,:,1]<=crop[3]))
    valid=np.isfinite(hits[:,:,0]);coverage=float(overlap[valid].mean())
    ok,encoded=cv2.imencode('.jpg',view,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:raise ValueError('Cannot encode DOP view preview')
    return encoded.tobytes(),coverage,actual


def height_view_preview(tile,points,height,frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from scipy.ndimage import distance_transform_edt
    cfg=yaml.safe_load(tile['config'].read_text());cfg['sequence']['frames']=int(frames);ref=Path(cfg['map']['reference_scene'])
    offset=np.asarray(tile['utm_offset'],float);x0,y0,x1,y1=tile['bounds_local']
    points=np.asarray(points,float);world_xy=np.column_stack((x0+points[:,0]*(x1-x0),y1-points[:,1]*(y1-y0)))+offset[:2]
    route=preview_route(world_xy,np.asarray(tile['bounds_local'])+np.tile(offset[:2],2),
                        cfg['sequence']['frames'],cfg['sequence']['fps'],cfg['trajectory'])
    loader=SceneLoader(ref,load_dsm=True,load_mesh=False);dsm,dsm_bounds=loader.dsm()
    local_xy=route['xy']-offset[:2];roof=dsm_heights(dsm,np.asarray(dsm_bounds),local_xy)
    local_z=plan_altitude(roof,float(height)+cfg['map']['vertical_offset_m'],
        cfg['trajectory']['safe_distance_m'],cfg['trajectory']['max_vertical_speed_mps'],cfg['sequence']['fps'])
    key=str(ref)
    if key not in mesh_height_cache:
        with np.load(ref/'mesh.npz') as data:vertices=data['vertices'].astype(np.float32)
        xmin,ymin=vertices[:,:2].min(0);xmax,ymax=vertices[:,:2].max(0);nx=ny=450
        ix=np.clip(((vertices[:,0]-xmin)/(xmax-xmin)*nx).astype(int),0,nx-1)
        iy=np.clip(((vertices[:,1]-ymin)/(ymax-ymin)*ny).astype(int),0,ny-1)
        grid=np.full((ny,nx),-np.inf,np.float32);np.maximum.at(grid,(iy,ix),vertices[:,2])
        valid=np.isfinite(grid)&(grid>-np.inf)
        nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
        grid=grid[tuple(nearest)]
        mesh_height_cache[key]=(grid,(float(xmin),float(xmax),float(ymin),float(ymax)))
    grid,extent=mesh_height_cache[key]
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(local_xy,axis=0),axis=1))]
    fig,(ax,profile)=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1.25,1]})
    image=ax.imshow(grid,extent=extent,origin='lower',cmap='terrain',aspect='equal')
    seg=np.stack((np.column_stack((local_xy[:-1,0],local_xy[:-1,1])),
                  np.column_stack((local_xy[1:,0],local_xy[1:,1]))),axis=1)
    norm=Normalize(float(np.nanmin(grid)),float(max(np.nanmax(grid),np.max(local_z))))
    route_line=LineCollection(seg,cmap='plasma',norm=norm,linewidth=2.5);route_line.set_array((local_z[:-1]+local_z[1:])/2)
    ax.add_collection(route_line);ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_title('Full mesh height + UAV trajectory')
    ax.set_xlabel('Local X (m)');ax.set_ylabel('Local Y (m)');fig.colorbar(image,ax=ax,label='Mesh Z (m)',shrink=.75)
    fig.colorbar(route_line,ax=ax,label='UAV Z (m)',shrink=.75)
    profile.plot(distance,roof,color='#66502b',label='Mesh/DSM surface')
    profile.plot(distance,local_z,color='#d74726',label='UAV altitude')
    profile.fill_between(distance,roof,local_z,color='#ef9a66',alpha=.2)
    profile.set_title('Altitude along trajectory');profile.set_xlabel('Distance along route (m)');profile.set_ylabel('Local Z (m)');profile.legend()
    fig.tight_layout();from io import BytesIO
    output=BytesIO();fig.savefig(output,format='png',dpi=130);plt.close(fig)
    return output.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref-path',type=Path,required=True,
                        help='Reference scene root or one scene directory')
    parser.add_argument('--save-path',type=Path,
                        help='Tile output root or one scene directory')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    args.ref_path=args.ref_path.resolve();args.save_path=(args.save_path or args.ref_path).resolve()
    lock=Lock()
    render_lock=Lock()
    height_lock=Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data,content_type='application/json'):
            if not isinstance(data,bytes): data=json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<2000000:raise ValueError('Request is too large')
            return json.loads(self.rfile.read(length))

        def do_GET(self):
            path=unquote(urlparse(self.path).path)
            if path=='/':return self.reply(200,PAGE.read_bytes(),'text/html; charset=utf-8')
            if path=='/api/scenes':return self.reply(200,scene_names(args.ref_path))
            if path.startswith('/api/source/'):
                scene=path.rsplit('/',1)[-1];info=source_info(scene,args.ref_path)
                h,w=info['image'].shape[:2]
                return self.reply(200,{'width':w,'height':h,'year':info['year'],
                                       'agl':info['agl'],'tilt':info['tilt']})
            if path.startswith('/source-image/'):
                scene=path.rsplit('/',1)[-1];image=source_info(scene,args.ref_path)['image']
                h,w=image.shape[:2];scale=min(1.,1800/max(h,w))
                if scale<1:image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
                ok,encoded=cv2.imencode('.jpg',image)
                if not ok:raise ValueError('Cannot encode reference DOP preview')
                return self.reply(200,encoded.tobytes(),'image/jpeg')
            if path=='/api/tiles':
                return self.reply(200,[{k:v for k,v in tile.items() if k not in ('image','route','utm_offset','sequence','trajectory','config')}
                                       |{'saved':tile['route'].is_file(),
                                         'generated':all((tile['config'].parent/path).exists()
                                                         for path in ('video.mp4','poses.csv','depth/metadata.json'))}
                                       for tile in tile_catalog(args.save_path,args.ref_path).values()])
            kind,_,name=path.lstrip('/').partition('/')
            tile=tile_catalog(args.save_path,args.ref_path).get(name)
            if tile is None:return self.reply(404,{'error':'Unknown DOP tile'})
            if kind=='dop':return self.reply(200,tile['image'].read_bytes(),'image/jpeg')
            if kind=='route':
                if not tile['route'].is_file():return self.reply(404,{'error':'No route saved'})
                return self.reply(200,tile['route'].read_bytes())
            if kind=='preview-image':
                preview=tile['config'].parent/'preview'/'first_frame_rgb.png'
                return self.reply(200,preview.read_bytes(),'image/png')
            return self.reply(404,{'error':'Not found'})

        def do_POST(self):
            path=unquote(urlparse(self.path).path)
            try:
                submitted=self.body()
                if path=='/api/accept':
                    candidate=crop_candidate(submitted['scene'],submitted['rect'],args.ref_path)
                    with lock:
                        name,config_path=accept_candidate(candidate,args.save_path,args.ref_path)
                    return self.reply(200,{'name':name,'config':str(config_path)})
                kind,_,name=path.lstrip('/').partition('/')
                tile=tile_catalog(args.save_path,args.ref_path).get(name)
                if kind=='plan-preview' and tile is not None:
                    if not tile['route'].is_file():raise ValueError('Save a route before planning the preview')
                    if not render_lock.acquire(blocking=False):return self.reply(409,{'error':'A GPU render is already running'})
                    try:
                        result=subprocess.run([sys.executable,str(ROOT/'dataset_pipline/pipeline.py'),
                                               'preview','--config',str(tile['config'])],
                                              cwd=ROOT,capture_output=True,text=True)
                    finally:render_lock.release()
                    log=(result.stdout+'\n'+result.stderr).strip()[-3000:]
                    if result.returncode:return self.reply(500,{'error':log})
                    return self.reply(200,{'log':log,'scene':str(tile['config'].parent)})
                if kind=='dop-view' and tile is not None:
                    points=np.asarray(submitted['points'],float)
                    if points.ndim!=2 or points.shape[1]!=2 or len(points)<4 or not np.isfinite(points).all():
                        raise ValueError('Draw a route first to preview its DOP view')
                    image,coverage,bounds=dop_view_preview(tile,points,submitted['height_agl_m'],
                                                            submitted['tilt_from_nadir_deg'],
                                                            submitted.get('camera_center_weight',tile['camera_center_weight']),
                                                            submitted['frames'])
                    self.send_response(200);self.send_header('Content-Type','image/jpeg')
                    self.send_header('Content-Length',str(len(image)))
                    self.send_header('X-DOP-Coverage',f'{coverage:.3f}')
                    self.send_header('X-DOP-Bounds',','.join(f'{v:.1f}' for v in bounds))
                    self.end_headers();self.wfile.write(image);return
                if kind=='height-view' and tile is not None:
                    with height_lock:image=height_view_preview(tile,submitted['points'],submitted['height_agl_m'],submitted['frames'])
                    return self.reply(200,image,'image/png')
                if kind=='generate' and tile is not None:
                    if not tile['route'].is_file():raise ValueError('Save a route before generating')
                    save_rgb=bool(submitted.get('rgb',True));save_depth=bool(submitted.get('depth',True))
                    if not save_rgb and not save_depth:raise ValueError('Select RGB, depth, or both')
                    if not render_lock.acquire(blocking=False):return self.reply(409,{'error':'A render is already running'})
                    try:
                        command=[sys.executable,str(ROOT/'dataset_pipline/pipeline.py'),
                                 'generate','--config',str(tile['config'])]
                        if save_depth:command.append('--depth')
                        if not save_rgb:command.append('--no-rgb')
                        result=subprocess.run(command,
                                              cwd=ROOT,capture_output=True,text=True)
                    finally:
                        render_lock.release()
                    log=(result.stdout+'\n'+result.stderr).strip()[-3000:]
                    if result.returncode:return self.reply(500,{'error':log})
                    return self.reply(200,{'log':log,'scene':str(tile['config'].parent)})
                if kind not in ('route','route-preview') or tile is None:return self.reply(404,{'error':'Unknown DOP tile'})
                points=np.asarray(submitted['points'],float)
                if points.ndim!=2 or points.shape[1]!=2 or not 4<=len(points)<=20000:
                    raise ValueError('Draw a continuous route with at least four points')
                if not np.isfinite(points).all() or np.any(points<0) or np.any(points>1):
                    raise ValueError('Route points must stay inside the DOP')
                pixel=points*np.array([tile['width'],tile['height']])
                if np.linalg.norm(np.diff(pixel,axis=0),axis=1).sum()<20:
                    raise ValueError('Draw a longer route')
                x0,y0,x1,y1=tile['bounds_local']
                local=np.column_stack((x0+points[:,0]*(x1-x0),y1-points[:,1]*(y1-y0)))
                world=local+np.asarray(tile['utm_offset'][:2])
                world_bounds=np.asarray(tile['bounds_local'],float)+np.tile(np.asarray(tile['utm_offset'][:2]),2)
                frames=int(submitted.get('frames',tile['frames']))
                if frames<2 or frames>20000:raise ValueError('Frame count must be between 2 and 20000')
                result=preview_route(world,world_bounds,frames,tile['sequence']['fps'],
                                     tile['trajectory'])
                metrics={k:v for k,v in result.items() if k not in ('xy','smooth')}
                metrics['frames']=frames
                height=float(submitted.get('height_agl_m',tile['agl_m']))
                tilt=float(submitted.get('tilt_from_nadir_deg',tile['tilt_deg']))
                aim=float(submitted.get('camera_center_weight',tile['camera_center_weight']))
                if not tile['agl_min_m']<=height<=tile['agl_max_m']:
                    raise ValueError('Height must stay within the reference AGL range')
                if not 20<=tilt<=30:raise ValueError('Camera tilt must be between 20° and 30° from nadir')
                if not 0<=aim<=1:raise ValueError('Camera aim must be between path and tile center')
                smooth_local=result['smooth']-np.asarray(tile['utm_offset'][:2])
                smooth_norm=np.column_stack(((smooth_local[:,0]-x0)/(x1-x0),
                                             (y1-smooth_local[:,1])/(y1-y0)))
                smooth_pixel=smooth_norm*np.array([tile['width'],tile['height']])
                if kind=='route-preview':
                    return self.reply(200,{'points_pixel_xy':smooth_pixel.tolist(),'metrics':metrics})
                route={'name':name,'source':'hand_drawn_2d','status':'draft',
                       'image_size_px':[tile['width'],tile['height']],
                       'dop_bounds_local':tile['bounds_local'],'utm_offset':tile['utm_offset'],
                       'reference_agl_m':[tile['agl_min_m'],tile['agl_m'],tile['agl_max_m']],
                       'reference_tilt_from_nadir_deg':tilt,'camera_center_weight':aim,
                       'frames':frames,
                       'points_normalized':points.tolist(),'points_pixel_xy':pixel.tolist(),
                       'points_local_xy_m':local.tolist(),'points_world_xy_m':world.tolist(),
                       'points_smoothed_normalized':smooth_norm.tolist(),
                       'points_smoothed_pixel_xy':smooth_pixel.tolist(),
                       'points_smoothed_world_xy_m':result['smooth'].tolist(),
                       'smoothing_m':result['smoothing_m'],'smooth_preview':metrics}
                target=tile['route'];target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(route,indent=2));tmp.replace(target)
                cfg=yaml.safe_load(tile['config'].read_text())
                cfg['trajectory']['height_agl_m']=height
                cfg['trajectory']['tilt_from_nadir_deg']=tilt
                cfg['trajectory']['camera_center_weight']=aim
                cfg['sequence']['frames']=frames
                temp=tile['config'].with_suffix('.yaml.tmp')
                temp.write_text(yaml.safe_dump(cfg,sort_keys=False));temp.replace(tile['config'])
                state_path=args.save_path/'selected_tiles.json'
                if state_path.is_file():
                    with lock:
                        state=json.loads(state_path.read_text())
                        if name in state.get('tiles',{}):
                            state['tiles'][name]['status']='drawn'
                            temp=state_path.with_suffix('.json.tmp')
                            temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
                return self.reply(200,{'saved':str(target),'points':len(points),
                                       'smooth_points':len(smooth_pixel),'points_pixel_xy':smooth_pixel.tolist(),
                                       'frames':frames,
                                       'metrics':metrics})
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
                return self.reply(400,{'error':str(error)})
            except Exception as error:
                traceback.print_exc()
                return self.reply(500,{'error':str(error)})

    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'Route editor: http://{args.host}:{args.port} ({len(scene_names(args.ref_path))} scenes)',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
