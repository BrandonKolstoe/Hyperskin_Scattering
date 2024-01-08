import os
import glob

import torch
import cv2
import h5py
import pickle
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

from hsi import HSIDataset
from typing import Any, Callable, Dict, List, Optional, Tuple


class Load_rgbvis(HSIDataset):
    resolution = {
        'height': 1024,
        'width': 1024,
        'bands': 31,
    }

    def __init__(self,
                rgb_dir: str,
                hsi_dir: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None):
        super().__init__(root = rgb_dir, transform=transform, target_transform=target_transform)

        # files location
        self.rgb_dir = rgb_dir
        self.hsi_dir = hsi_dir
        self.rgb_files = sorted(glob.glob(self.rgb_dir + "/*.jpg"))
        self.cube_files = sorted(glob.glob(self.hsi_dir + "/*.mat"))

        # total data
        self.rgb_files = np.asarray(self.rgb_files)
        self.cube_files = np.asarray(self.cube_files)
        self.total_files = len(self.rgb_files)

    def loadCube(self, cube_path):
        '''
        return cube in (h, w, c=31)
        range: (0, 1)
        '''
        with h5py.File(cube_path, 'r') as f:
            cube = np.squeeze(np.float32(np.array(f['cube'])))
            cube = np.transpose(cube, [2,1,0])
            f.close()
        return cube

    def loadData(self, img_path, cube_path):
        # load image file
        rgb = plt.imread(img_path)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        # load cube file
        cube = self.loadCube(cube_path)

        return rgb, cube

    def __getitem__(self, idx):
        rgb, cube = self.loadData(self.rgb_files[idx], self.cube_files[idx])

        if self.transform is not None:
            all = np.concatenate([rgb, cube], axis = -1)
            all = self.transform(all)
            rgb = all[:3, :, :]
            cube = all[3:, :, :]

        return rgb, cube

    def __len__(self):
        return self.total_files

class Load_msinir(HSIDataset):
    resolution = {
        'height': 1024,
        'width': 1024,
        'bands': 31,
    }

    def __init__(self,
                msi_dir: str,
                hsi_dir: str,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None):
        super().__init__(root = msi_dir, transform=transform, target_transform=target_transform)

        # files location
        self.msi_dir = msi_dir
        self.hsi_dir = hsi_dir
        self.msi_files = sorted(glob.glob(self.msi_dir + "/*.mat"))
        self.cube_files = sorted(glob.glob(self.hsi_dir + "/*.mat"))

        # total data
        self.msi_files = np.asarray(self.msi_files)
        self.cube_files = np.asarray(self.cube_files)
        self.total_files = len(self.msi_files)


    def loadCube(self, cube_path):
        '''
        return cube in (h, w, c=31)
        range: (0, 1)
        '''
        with h5py.File(cube_path, 'r') as f:
            cube = np.squeeze(np.float32(np.array(f['cube'])))
            cube = np.transpose(cube, [2,1,0])
            f.close()
        return cube

    def loadData(self, img_path, cube_path):
        # load MSI data (RGB + 960nm) 1024*1024*4
        msi = self.loadCube(img_path)

        # load cube file
        cube = self.loadCube(cube_path)

        return msi, cube

    def __getitem__(self, idx):
        msi, cube = self.loadData(self.msi_files[idx], self.cube_files[idx])

        if self.transform is not None:
            all = np.concatenate([msi, cube], axis = -1)
            all = self.transform(all)
            msi = all[:4, :, :]
            cube = all[4:, :, :]

        return msi, cube

    def __len__(self):
        return self.total_files


class Load_msi_visnir(HSIDataset):
    resolution = {
        'height': 1024,
        'width': 1024,
        'bands': 62,
    }

    def __init__(self,
                msi_dir: str,
                vis_dir: str,
                nir_dir: str,
                train_valid_mask: bool = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None):
        super().__init__(root = msi_dir, transform=transform, target_transform=target_transform)

        # files location
        self.msi_dir = msi_dir
        self.vis_dir = vis_dir
        self.nir_dir = nir_dir
        self.msi_files = np.asarray(sorted(glob.glob(self.msi_dir + "/*.mat")))
        self.vis_files = np.asarray(sorted(glob.glob(self.vis_dir + "/*.mat")))
        self.nir_files = np.asarray(sorted(glob.glob(self.nir_dir + "/*.mat")))

        # total data
        if train_valid_mask is not None:
            self.msi_files = self.msi_files[train_valid_mask]
            self.cube_files = self.cube_files[train_valid_mask]
        self.total_files = len(self.msi_files)


    def loadCube(self, cube_path):
        '''
        return cube in (h, w, c=31)
        range: (0, 1)
        '''
        with h5py.File(cube_path, 'r') as f:
            cube = np.squeeze(np.float32(np.array(f['cube'])))
            cube = np.transpose(cube, [2,1,0])
            f.close()
        return cube

    def loadData(self, img_path, vis_path, nir_path):
        # load MSI data (RGB + 960nm) 1024*1024*4
        #print('hi')
        msi = self.loadCube(img_path)

        # load cube file (vis + nir)
        vis = self.loadCube(vis_path)
        nir = self.loadCube(nir_path)
        cube = np.concatenate([vis, nir], axis = -1)

        return msi, cube

    def __getitem__(self, idx):
        msi, cube = self.loadData(self.msi_files[idx], self.vis_files[idx], self.nir_files[idx])

        if self.transform is not None:
            all = np.concatenate([msi, cube], axis = -1)
            all = self.transform(all)
            msi = all[:4, :, :]
            cube = all[4:, :, :]

        return msi, cube

    def __len__(self):
        return self.total_files
