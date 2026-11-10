"""Cached access to the reference data used by one tile."""
import json
import csv
from pathlib import Path
import shutil

import cv2
import numpy as np

class SceneLoader:
    def __init__(self, reference_dir, load_dsm=True, load_mesh=True):
        self.root=Path(reference_dir)
        self.meta=json.loads((self.root/'meta.json').read_text())
        self._dsm=None
        self._mesh=None
        if load_dsm:
            with np.load(self.root/'dsm.npz') as data:
                self._dsm=data['height'].copy(),data['bounds'].copy()
        if load_mesh:
            with np.load(self.root/'mesh.npz') as data:
                self._mesh=data['vertices'].astype(np.float32),data['faces'].astype(np.uint32)
        self._dops={}

    def dsm(self):
        return self._dsm

    def mesh(self):
        return self._mesh

    def pose_table(self):
        with (self.root/'poses.csv').open(newline='') as f:
            return list(csv.DictReader(f))

    def dop_info(self, year=None, render=False):
        if render and self.meta.get('render_dop'):
            return self.meta['render_dop']
        year=year or max(int(k.removeprefix('dop_')) for k in self.meta['dops'])
        return self.meta['dops'][f'dop_{year}']

    def dop(self, year=None, render=False):
        info=self.dop_info(year,render)
        key=info['file']
        if key not in self._dops:
            image=cv2.imread(str(self.root/key))
            if image is None: raise FileNotFoundError(self.root/key)
            self._dops[key]=image
        return info,self._dops[key]


def create_output_dirs(scene, depth=False, clear_depth=False):
    scene=Path(scene)
    for name in ('dop','preview'):
        (scene/name).mkdir(parents=True,exist_ok=True)
    if depth:
        if clear_depth and (scene/'depth').exists():shutil.rmtree(scene/'depth')
        (scene/'depth').mkdir(parents=True,exist_ok=True)
    return scene
