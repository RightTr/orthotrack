"""Browser UI for rerolling DOP crops and drawing routes on accepted tiles."""
import argparse
import json
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
from dataset_pipline.route_editor.tiles import DATA,OUT,accept_candidate,crop_candidate,scene_names,source_info
from dataset_pipline.loader import SceneLoader
from dataset_pipline.planner import preview_route

PAGE=Path(__file__).with_name('index.html')


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
                       'agl_min_m':cfg['trajectory']['reference_agl_min_m'],
                       'agl_m':cfg['trajectory']['reference_agl_m'],
                       'agl_max_m':cfg['trajectory']['reference_agl_max_m'],
                       'tilt_deg':cfg['trajectory']['tilt_from_nadir_deg'],
                       'sequence':cfg['sequence'],'trajectory':cfg['trajectory']}
    return catalog


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref-path',type=Path,default=DATA,
                        help='MovingDrone_Ref root or one scene directory')
    parser.add_argument('--save-path',type=Path,default=OUT,
                        help='Sequence root or one sequence directory; creates tileXX inside the matching sequence')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    args.ref_path=args.ref_path.resolve();args.save_path=args.save_path.resolve()
    lock=Lock()

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
                return self.reply(200,[{k:v for k,v in tile.items() if k not in ('image','route','utm_offset','bounds_local','sequence','trajectory')}
                                       |{'saved':tile['route'].is_file()} for tile in tile_catalog(args.save_path,args.ref_path).values()])
            kind,_,name=path.lstrip('/').partition('/')
            tile=tile_catalog(args.save_path,args.ref_path).get(name)
            if tile is None:return self.reply(404,{'error':'Unknown DOP tile'})
            if kind=='dop':return self.reply(200,tile['image'].read_bytes(),'image/jpeg')
            if kind=='route':
                if not tile['route'].is_file():return self.reply(404,{'error':'No route saved'})
                return self.reply(200,tile['route'].read_bytes())
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
                result=preview_route(world,world_bounds,tile['sequence']['frames'],tile['sequence']['fps'],
                                     tile['trajectory'])
                smooth_local=result['smooth']-np.asarray(tile['utm_offset'][:2])
                smooth_norm=np.column_stack(((smooth_local[:,0]-x0)/(x1-x0),
                                             (y1-smooth_local[:,1])/(y1-y0)))
                smooth_pixel=smooth_norm*np.array([tile['width'],tile['height']])
                metrics={k:v for k,v in result.items() if k not in ('xy','smooth')}
                if kind=='route-preview':
                    return self.reply(200,{'points_pixel_xy':smooth_pixel.tolist(),'metrics':metrics})
                route={'name':name,'source':'hand_drawn_2d','status':'draft',
                       'image_size_px':[tile['width'],tile['height']],
                       'dop_bounds_local':tile['bounds_local'],'utm_offset':tile['utm_offset'],
                       'reference_agl_m':[tile['agl_min_m'],tile['agl_m'],tile['agl_max_m']],
                       'reference_tilt_from_nadir_deg':tile['tilt_deg'],
                       'points_normalized':points.tolist(),'points_pixel_xy':pixel.tolist(),
                       'points_local_xy_m':local.tolist(),'points_world_xy_m':world.tolist(),
                       'points_smoothed_normalized':smooth_norm.tolist(),
                       'points_smoothed_pixel_xy':smooth_pixel.tolist(),
                       'points_smoothed_world_xy_m':result['smooth'].tolist(),
                       'smoothing_m':result['smoothing_m'],'smooth_preview':metrics}
                target=tile['route'];target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(route,indent=2));tmp.replace(target)
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
