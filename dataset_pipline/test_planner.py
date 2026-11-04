"""Fast planner checks without map downloads or rendering."""
from pathlib import Path
import unittest

import numpy as np
import yaml

from dataset_pipline.planner import make_route
from dataset_pipline.trajectory import make_poses


class ForwardSweepTest(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        self.cfg['sequence']['frames']=3000
        self.cfg['sequence']['fov_vertical_deg']=26.9
        self.bounds=np.array([-400.,-320.,400.,320.])

    def test_route_is_reproducible_and_continuous(self):
        a=make_route(self.cfg,np.zeros(2),0.,self.bounds)
        b=make_route(self.cfg,np.zeros(2),0.,self.bounds)
        self.assertTrue(np.array_equal(a,b))
        self.assertEqual(a.shape,(3000,3))
        self.assertTrue(np.isfinite(a).all())
        self.assertLess(np.linalg.norm(np.diff(a,axis=0),axis=1).max()*15,15.15)
        rotations=make_poses(a,self.cfg)
        self.assertTrue(np.isfinite(rotations).all())


if __name__=='__main__': unittest.main()
