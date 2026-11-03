"""Fast route checks without map downloads or rendering."""
import copy
from pathlib import Path
import unittest

import numpy as np
import yaml

from dataset_pipline.planner import make_route
from dataset_pipline.trajectory import make_poses


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        self.cfg['trajectory']['height_agl_m']=208
        self.cfg['trajectory']['height_variation_m']=208*self.cfg['trajectory']['height_variation_ratio']
        self.bounds=np.array([-918,-761,918,761],float)

    def test_repeatable_long_variable_route(self):
        a=make_route(self.cfg,np.zeros(2),0,self.bounds)
        self.assertTrue(np.array_equal(a,make_route(self.cfg,np.zeros(2),0,self.bounds)))
        other=copy.deepcopy(self.cfg);other['trajectory']['seed']+=1
        self.assertGreater(np.max(np.abs(a-make_route(other,np.zeros(2),0,self.bounds))),1)
        distance=np.linalg.norm(np.diff(a,axis=0),axis=1).sum()
        speed=np.linalg.norm(np.diff(a,axis=0),axis=1).max()*self.cfg['sequence']['fps']
        self.assertGreater(distance,3000)
        self.assertLess(speed,self.cfg['trajectory']['max_speed_mps'])
        self.assertGreater(np.ptp(a[:,2]),90)
        heading=np.unwrap(np.arctan2(np.gradient(a[:,1]),np.gradient(a[:,0])))
        self.assertGreater(np.abs(np.diff(heading)).sum(),2*np.pi)
        rotation=make_poses(a,self.cfg)
        self.assertTrue(np.isfinite(rotation).all())
        self.assertGreater(np.ptp(np.rad2deg(np.arccos(-rotation[:,2,2]))),10)

    def test_route_shapes_and_fixed_height(self):
        routes=[]
        for shape in ('straight','curved','crossing'):
            cfg=copy.deepcopy(self.cfg);cfg['trajectory']['route_shape']=shape
            cfg['trajectory']['height_variation_m']=0
            positions=make_route(cfg,np.zeros(2),0,self.bounds)
            self.assertLess(np.ptp(positions[:,2]),1e-10)
            routes.append(positions)
        self.assertGreater(np.max(np.abs(routes[0][:,:2]-routes[1][:,:2])),20)
        self.assertGreater(np.max(np.abs(routes[1][:,:2]-routes[2][:,:2])),20)

    def test_ablations_change_one_parameter(self):
        self.assertEqual(self.cfg['ablations']['baseline'],{})
        self.assertNotIn('path_more_turns',self.cfg['ablations'])
        for name,change in self.cfg['ablations'].items():
            if name=='baseline': continue
            self.assertEqual(len(change),1,name)
            self.assertIn(next(iter(change)),self.cfg['trajectory'])


if __name__=='__main__': unittest.main()
