#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 11:11:41 2023

@author: whitaker
"""

import numpy as np
from PIL import Image
import cv2,os
from scipy import ndimage
import scipy.io
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from os import listdir
from os.path import isfile, join

from skimage.util import random_noise
from skimage import feature

import cv2,time


def load_fluence():
    path = '/media/whitaker-160/bigstorage/NeRF/Data/Fluence.mat'
    mat = scipy.io.loadmat(path)['flu_resize']  
    mat = mat/np.max(mat)
    return np.array(mat).T

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def int_to_binary_array(arr):
    """Converts an array of integers to binary arrays with 8-bit length."""
    binary_arrays = np.zeros((len(arr), 13), dtype=np.uint8)
    for i, n in enumerate(arr):
        binary_str = bin(n)[2:].zfill(13)  # Convert n to a binary string with 8-bit length
        binary_arrays[i] = np.array([int(b) for b in binary_str], dtype=np.uint8)
    return binary_arrays

def pos_to_freq(pos, freq_max=10):
    """
    Transforms a 6-dimensional position vector to the frequency domain using sine and cosine functions.
    """
    assert pos.shape[1] == 6,"Input position vector must have shape (6,)"
    batch_size = pos.shape[0]
    # Compute the sine and cosine values for each frequency
    freqs = np.arange(1, freq_max + 1)
    sin_cos = np.zeros((batch_size, 2 * freq_max*6))
    for i, freq in enumerate(freqs):
        sin_cos[:, 12*i:12*i+6] = np.sin(freq * np.pi * pos)
        sin_cos[:, 12*i+6:12*i+12] = np.cos(freq * np.pi * pos)
    
    # Concatenate the sine and cosine values for all frequencies

    return sin_cos

class PAT_3D_train_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, Simu_files, phanU_files,phanS_files, transform=None):
        """
        Args:
            Bscans (string): Path to the Bscans file.
            angles (string): Angle for each Bscan.
            ground_truth (string): Path to the ground_truth file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        Bscans,us=[],[]
        Bscan_loc,us_loc = Simu_files+'Bscan/',Simu_files+'US/'
        Bscan_files = [join(Bscan_loc, f) for f in sorted(listdir(Bscan_loc)) if isfile(join(Bscan_loc, f))]

        us_files = [join(us_loc, f) for f in sorted(listdir(us_loc)) if isfile(join(us_loc, f))]
        
        for file_loc in Bscan_files:
            mat = scipy.io.loadmat(file_loc)['Bscan']            
            Bscans.append(mat)

        for file_loc in us_files:
            mat = scipy.io.loadmat(file_loc)['us_resize']            
            us.append(mat)     
        
        
        Bscans_phanS,us_phanS=[],[]
        Bscan_loc_phanS,us_loc_phanS = phanS_files+'Bscan/',phanS_files+'US/'
        Bscan_files_phanS = [join(Bscan_loc_phanS, f) for f in sorted(listdir(Bscan_loc_phanS)) if isfile(join(Bscan_loc_phanS, f))]
        us_files_phanS = [join(us_loc_phanS, f) for f in sorted(listdir(us_loc_phanS)) if isfile(join(us_loc_phanS, f))]
        
        for file_loc in Bscan_files_phanS:
            mat = scipy.io.loadmat(file_loc)['pa_resize']            
            Bscans_phanS.append(mat)
        for file_loc in us_files_phanS:
            mat = scipy.io.loadmat(file_loc)['us_resize']            
            us_phanS.append(mat)     
            
        Bscans_phanU,us_phanU=[],[]
        Bscan_loc_phanU,us_loc_phanU = phanU_files+'Bscan/',phanU_files+'US/'
        Bscan_files_phanU = [join(Bscan_loc_phanU, f) for f in sorted(listdir(Bscan_loc_phanU)) if isfile(join(Bscan_loc_phanU, f))]
        us_files_phanU = [join(us_loc_phanU, f) for f in sorted(listdir(us_loc_phanU)) if isfile(join(us_loc_phanU, f))]
        
        for file_loc in Bscan_files_phanU:
            mat = scipy.io.loadmat(file_loc)['pa_resize']            
            Bscans_phanU.append(mat)
        for file_loc in us_files_phanU:
            mat = scipy.io.loadmat(file_loc)['us_resize']            
            us_phanU.append(mat)     
            
            
        ## Load Clinical images    
        # Files_PAT=[]
        # Files_PE=[]
        # BSCAN_ID = {}
        # for root, dirs, files in os.walk('/media/whitaker-160/bigstorage/NeRF/Data/Clinical/P10/', topdown=True):
        #     files.sort()
        #     for name in files:

        #         if '_PAT_' in name: 
        #             BSCAN_ID[name[-10:-4]]=os.path.join(root, name)

        #         elif '_PE_' in name:
        #             if name[-10:-4] in BSCAN_ID.keys():
        #                 Files_PE.append(os.path.join(root, name))
        #                 Files_PAT.append(BSCAN_ID[name[-10:-4]])
        # clinical_Bscans=[]
        # for file_loc in Files_PAT:
        #     mat = scipy.io.loadmat(file_loc)['file']
        #     mat[mat==255]=0
        #     mat = mat/np.amax(mat)*255
        #     mat = np.array(mat, dtype='uint8')
        #     mat = cv2.resize(mat,(128,128),interpolation=cv2.INTER_LINEAR)
            

        #     # dynamic range
        #     min_dB = 10**(-10/20)
        #     view_PAT_log = mat/np.max(np.max((mat)))
        #     idx = view_PAT_log < min_dB
        #     view_PAT_log = 255*((20/10)*np.log10(view_PAT_log+1e-6)+1)
        #     view_PAT_log[idx] = 0
            
        #     clinical_Bscans.append(view_PAT_log)
            
        # clinical_US_imgs = []
        # for file_loc in Files_PE:
        #     mat = scipy.io.loadmat(file_loc)['file']
        #     mat[mat==255]=0
        #     mat = mat/np.amax(mat)*255
        #     mat = np.array(mat, dtype='uint8')
        #     mat = cv2.resize(mat,(128,128),interpolation=cv2.INTER_LINEAR)

        #     # dynamic range
        #     min_dB = 10**(-50/20)
        #     view_PE_log = mat/np.max(np.max((mat)))
        #     idx = view_PE_log < min_dB
        #     view_PE_log = 255*((20/50)*np.log10(view_PE_log+1e-6)+1)
        #     view_PE_log[idx] = 0
            
        #     clinical_US_imgs.append(view_PE_log)
            
        
        ## washu
        
        # new_idx = [4,0,2,1,3]
        # slices=[72,74,76,78,80,82,84,86,88]
        # self.Bscans = np.array(Bscans[:72]+[Bscans[i] for i in slices])
        # self.us = np.array(us[:72]+[us[i] for i in slices])
        # self.angles = np.array([i for i in range(0,180,10)]*4+[0,20,40,60,80,100,120,140,160])
        
        new_idx = [4,0,2,1,3]
        self.Bscans = np.concatenate([np.array(Bscans),np.array(Bscans_phanU),np.array(Bscans_phanS)[[0,6,12,17]]], axis=0)
        self.us = np.concatenate([np.array(us),np.array(us_phanU),np.array(us_phanS)[[0,6,12,17]]], axis=0)
        self.angles = np.array([i for i in range(0,180,10)]*6+[0,60,120,170])
        
        # self.Bscans = np.concatenate([np.array(Bscans_phanS),np.array(Bscans_phanU)[[0]]], axis=0)
        # self.us = np.concatenate([np.array(us_phanS),np.array(us_phanU)[[0]]], axis=0)
        # self.angles = np.array([i for i in range(0,180,10)]+[0])
        
        # self.Bscans = np.array(Bscans_phanU)[[0,3,6,9,12]]
        # self.us = np.array(us_phanU)[[0,3,6,9,12]]
        # self.angles = np.array([0,30,60,90,120])
        
        # self.Bscans = np.array(clinical_Bscans)
        # self.us = np.array(clinical_US_imgs)
        # self.angles = np.array([0,90])
        
        # self.fluence = np.array(fluence)
        # self.gt = np.array(gt)
        # self.gt = self.gt[new_idx]



    def __len__(self):
        #print(len(self.Bscans))
        return len(self.Bscans)

    def __getitem__(self, idx):
        # input was x,y,z,US_value,Bscan_value,angle output is fluence, ua
        if torch.is_tensor(idx):
            idx = idx.tolist()
        bscan_idx = idx
        angle_idx = idx
        angle=self.angles[angle_idx]
        angle_rad = angle*np.pi/180
        letter = (bscan_idx)//18
        
        #pixel 2 pixel
        # nx, nz = (128, 128)
        # x,z = np.linspace(0, 127, nx),np.linspace(0, 127, nz)
        # xv, zv= np.meshgrid(x, z)
        # xv, zv=xv.flatten().reshape(-1,1).astype(int), zv.flatten().reshape(-1,1).astype(int)

        
        # ele_x=((64+(zv-64)*np.cos(angle_rad))%128).astype(int)
        # ele_y=((64-(zv-64)*np.sin(angle_rad))%128).astype(int)
        # ele_z = np.array(128**2*[0]).reshape(-1,1)
        # depth = xv
        
        # bscan_pixel = self.Bscans[idx][zv,xv]
        # us_pixel = self.us[idx][zv,xv]
        # identifier = np.array([0]*18+[1]*18+[2]*18+[3]*18+[4]*18+[5]*(len(self.Bscans)-90))/5
        # ID = np.array([identifier[idx]]*(128**2)).reshape(-1,1)
        # #ID = np.tile(int_to_binary_array(np.array([identifier[idx]])), (128*128, 1))
        # angles=np.array(128**2*[angle_rad]).reshape(-1,1)        
        
        # pixel 2 ray
        nx = (128)
        x = np.linspace(0, 127, nx).astype(int)       
        ele_x=((64+(x-64)*np.cos(angle_rad))%128).astype(int).reshape(-1,1) 
        ele_y=((64-(x-64)*np.sin(angle_rad))%128).astype(int).reshape(-1,1) 
        ele_z = np.array([1]*128).reshape(-1,1) 
        depth = np.array([0]*128).reshape(-1,1) 
        
        bscan_pixel = self.Bscans[idx][x].reshape(-1,128)  
        us_pixel = self.us[idx][x].reshape(-1,128)  
        
        ## angles
        angles=np.array(128*[angle_rad]).reshape(-1,1)  
        
        # identifier = np.array([0]*18+[1]*18+[2]*18+[3]*18+[4]*9)/6
        # ID = np.array([identifier[idx]]*(128)).reshape(-1,1)
        
        # identifier = (np.array(list(range(int(len(self.Bscans)/2)))*2)+6)/6
        
        # ID = np.array([identifier[idx]]*(128)).reshape(-1,1)
        
        identifier = np.array([0]*18+[1]*18+[2]*18+[3]*18+[4]*18+[5]*18+[6]*4)/6
        ID = np.array([identifier[idx]]*(128)).reshape(-1,1)
        #flue_pixel=self.fluence[bscan_idx][ele_x,ele_y,ele_z]
        #ua_pixel = self.gt[letter][ele_x,ele_y,ele_z]

        
        scale_pos = np.array(128).astype(float)
        scale_data = np.array(255).astype(float)
        
        input_data = np.concatenate([ele_x/scale_pos,ele_y/scale_pos,ele_z/scale_pos,depth/scale_pos,angles,ID],axis=1)
        input_data = pos_to_freq(input_data)
        output_data = np.concatenate([us_pixel/scale_data,bscan_pixel/scale_data],axis=1)
        
        return (input_data,output_data)
        
        
class PAT_3D_test_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, Simu_files, simu_idx, transform=None):
        """
        Args:
            Bscans (string): Path to the Bscans file.
            angles (string): Angle for each Bscan.
            ground_truth (string): Path to the ground_truth file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Bscans,fluence,gt,us=[],[],[],[]
        # Bscan_loc,fluence_loc,gt_loc,us_loc = Simu_files+'Bscan/',Simu_files+'Fluence/',Simu_files+'GT/',Simu_files+'US/'
        # Bscan_files = [join(Bscan_loc, f) for f in sorted(listdir(Bscan_loc)) if isfile(join(Bscan_loc, f))]
        # fluence_files = [join(fluence_loc, f) for f in sorted(listdir(fluence_loc)) if isfile(join(fluence_loc, f))]
        # gt_files = [join(gt_loc, f) for f in sorted(listdir(gt_loc)) if isfile(join(gt_loc, f))]
        # us_files = [join(us_loc, f) for f in sorted(listdir(us_loc)) if isfile(join(us_loc, f))]
        
        # for file_loc in Bscan_files:
        #     mat = scipy.io.loadmat(file_loc)['Bscan']            
        #     Bscans.append(mat)
        # for file_loc in fluence_files:
        #     mat = scipy.io.loadmat(file_loc)['fluence_resize']            
        #     fluence.append(mat)
            
        # # A, H, S, U, W
        # for file_loc in gt_files:
        #     mat = scipy.io.loadmat(file_loc)['gt']            
        #     gt.append(mat)
        # for file_loc in us_files:
        #     mat = scipy.io.loadmat(file_loc)['us_resize']            
        #     us.append(mat)      
        
        Files_PAT=[]
        Files_PE=[]
        BSCAN_ID = {}
        for root, dirs, files in os.walk('/media/whitaker-160/bigstorage/NeRF/Data/Clinical/P10/', topdown=True):
            files.sort()
            for name in files:
                
                if '_PAT_' in name: 
                    BSCAN_ID[name[-10:-4]]=os.path.join(root, name)
    
                elif '_PE_' in name:
                    if name[-10:-4] in BSCAN_ID.keys():
                        Files_PE.append(os.path.join(root, name))
                        Files_PAT.append(BSCAN_ID[name[-10:-4]])
        clinical_Bscans=[]
        for file_loc in Files_PAT[simu_idx:simu_idx+1]:
            mat = scipy.io.loadmat(file_loc)['file']
            mat[mat==255]=0
            mat = mat/np.amax(mat)*255
            mat = np.array(mat, dtype='uint8')
            mat = cv2.resize(mat,(128,128),interpolation=cv2.INTER_LINEAR)


            
            min_dB = 10**(-10/20)
            view_PAT_log = mat/np.max(np.max((mat)))
            idx = view_PAT_log < min_dB
            view_PAT_log = 255*((20/10)*np.log10(view_PAT_log+1e-6)+1)
            view_PAT_log[idx] = 0

            clinical_Bscans.extend([view_PAT_log]*180)

        clinical_US_imgs = []
        for file_loc in Files_PE[simu_idx:simu_idx+1]:
            mat = scipy.io.loadmat(file_loc)['file']
            mat[mat==255]=0
            mat = mat/np.amax(mat)*255
            mat = np.array(mat, dtype='uint8')
            mat = cv2.resize(mat,(128,128),interpolation=cv2.INTER_LINEAR)

            # dynamic range
            min_dB = 10**(-50/20)
            view_PE_log = mat/np.max(np.max((mat)))
            idx = view_PE_log < min_dB
            view_PE_log = 255*((20/50)*np.log10(view_PE_log+1e-6)+1)
            view_PE_log[idx] = 0
    
            clinical_US_imgs.extend([view_PE_log]*180)
       
    
        ## washu
        new_idx = [4,0,2,1,3]
        self.simu_idx =simu_idx
        # self.Bscans = np.concatenate([np.array(Bscans),np.array(clinical_Bscans)], axis=0)[simu_idx:simu_idx+1]
        # self.us = np.concatenate([np.array(us),np.array(clinical_US_imgs)], axis=0)[simu_idx:simu_idx+1]
        # self.angles = np.array([i for i in range(0,180,10)]*5+ [0,60]*len(clinical_US_imgs))[simu_idx:simu_idx+1]
        self.angles = np.array([i for i in range(0,180,1)]*int(len(clinical_Bscans)/180))
        self.Bscans = np.array(clinical_Bscans)
        self.us = np.array(clinical_US_imgs)
        
        
        #self.fluence = np.array(fluence)
        #self.gt = np.array(gt)
        #self.gt = self.gt[new_idx]
        #self.gt = self.gt[simu_idx//18:simu_idx//18+1]



    def __len__(self):

        return len(self.Bscans)

    def __getitem__(self, idx):
        # input was x,y,z,US_value,Bscan_value,angle output is fluence, ua
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        bscan_idx = self.simu_idx
        angle_idx = bscan_idx%180
        angle=self.angles[idx]
        angle_rad = angle*np.pi/180
        
        # # pixel 2 pixel
        # nx, nz = (128, 128)
        # x,z = np.linspace(0, 127, nx),np.linspace(0, 127, nz)
        # xv, zv= np.meshgrid(x, z)
        # xv, zv=xv.flatten().reshape(-1,1).astype(int), zv.flatten().reshape(-1,1).astype(int)
        

        
        # ele_x=((64+(zv-64)*np.cos(angle_rad))%128).astype(int)
        # ele_y=((64-(zv-64)*np.sin(angle_rad))%128).astype(int)
        # ele_z = np.array(128**2*[0]).reshape(-1,1)
        # depth = xv
        
        # bscan_pixel = self.Bscans[idx][zv,xv]
        # us_pixel = self.us[idx][zv,xv]
        # identifier = np.array([0]*18+[1]*18+[2]*18+[3]*18+[4]*18+[5]*(len(self.Bscans)-90))/5
        # ID = np.array([identifier[idx]]*(128**2)).reshape(-1,1)
        # #ID = np.tile(int_to_binary_array(np.array([identifier[idx]])), (128*128, 1))
        # angles=np.array(128**2*[angle_rad]).reshape(-1,1)
        
        # pixel 2 ray
        nx = (128)
        x = np.linspace(0, 127, nx).astype(int)

        
        ele_x=((64+(x-64)*np.cos(angle_rad))%128).astype(int).reshape(-1,1) 
        ele_y=((64-(x-64)*np.sin(angle_rad))%128).astype(int).reshape(-1,1) 
        ele_z = np.array([1]*128).reshape(-1,1) 
        depth = np.array([0]*128).reshape(-1,1) 
        
        bscan_pixel = self.Bscans[idx][x].reshape(-1,128)  
        us_pixel = self.us[idx][x].reshape(-1,128)  
        
        
        ## ID
        
        identifier = (np.array([i//180 for i in range(len(self.Bscans))])+6)/6
        ID = np.array([identifier[idx]]*(128)).reshape(-1,1)
        #ID = np.tile(int_to_binary_array(np.array([identifier[idx]])), (128*128, 1))
        angles=np.array(128*[angle_rad]).reshape(-1,1)
        
        scale_pos = np.array(128).astype(float)
        scale_data = np.array(255).astype(float)
                
        
        input_data = np.concatenate([ele_x/scale_pos,ele_y/scale_pos,ele_z/scale_pos,depth/scale_pos,angles,ID], axis=1)
        input_data = pos_to_freq(input_data)
        output_data = np.concatenate([us_pixel/scale_data,bscan_pixel/scale_data], axis=1)
        
        return (input_data, output_data)

    
class PAT_3D_validate_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, Simu_files, simu_idx, transform=None):
        """
        Args:
            Bscans (string): Path to the Bscans file.
            angles (string): Angle for each Bscan.
            ground_truth (string): Path to the ground_truth file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # Bscans,us=[],[]
        # Bscan_loc,us_loc = Simu_files+'Bscan/',Simu_files+'US/'
        # Bscan_files = [join(Bscan_loc, f) for f in sorted(listdir(Bscan_loc)) if isfile(join(Bscan_loc, f))]
        # us_files = [join(us_loc, f) for f in sorted(listdir(us_loc)) if isfile(join(us_loc, f))]
        
        # for file_loc in Bscan_files:
        #     mat = scipy.io.loadmat(file_loc)['Bscan']            
        #     Bscans.append(mat)
        # for file_loc in us_files:
        #     mat = scipy.io.loadmat(file_loc)['us_resize']            
        #     us.append(mat)      
        
        Bscans_phanS,us_phanS=[],[]
        Bscan_loc_phanS,us_loc_phanS = Simu_files+'Bscan/',Simu_files+'US/'
        Bscan_files_phanS = [join(Bscan_loc_phanS, f) for f in sorted(listdir(Bscan_loc_phanS)) if isfile(join(Bscan_loc_phanS, f))]
        us_files_phanS = [join(us_loc_phanS, f) for f in sorted(listdir(us_loc_phanS)) if isfile(join(us_loc_phanS, f))]
        
        for file_loc in Bscan_files_phanS:
            mat = scipy.io.loadmat(file_loc)['pa_resize']            
            Bscans_phanS.append(mat)
            
        for file_loc in us_files_phanS:
            mat = scipy.io.loadmat(file_loc)['us_resize']            
            us_phanS.append(mat)
        ## washu
        self.simu_idx =simu_idx
        
        # self.Bscans = np.array(Bscans[72:])
        # self.us = np.array(us[72:])
        # self.angles = np.array([i for i in range(0,180,10)])       
        
        self.Bscans = np.array(Bscans_phanS)
        self.us = np.array(us_phanS)
        self.angles = np.array([i for i in range(0,180,10)])             
        
        #self.fluence = np.array(fluence)
        #self.gt = np.array(gt)
        #self.gt = self.gt[new_idx]
        #self.gt = self.gt[simu_idx//18:simu_idx//18+1]



    def __len__(self):
        return len(self.Bscans)

    def __getitem__(self, idx):
        # input was x,y,z,US_value,Bscan_value,angle output is fluence, ua
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        bscan_idx = self.simu_idx
        angle_idx = bscan_idx%18
        angle=self.angles[idx]
        angle_rad = angle*np.pi/180
        
        # pixel 2 ray
        nx = (128)
        x = np.linspace(0, 127, nx).astype(int)

        
        ele_x=((64+(x-64)*np.cos(angle_rad))%128).astype(int).reshape(-1,1) 
        ele_y=((64-(x-64)*np.sin(angle_rad))%128).astype(int).reshape(-1,1) 
        ele_z = np.array([1]*128).reshape(-1,1) 
        depth = np.array([0]*128).reshape(-1,1) 
        
        bscan_pixel = self.Bscans[idx][x].reshape(-1,128)  
        us_pixel = self.us[idx][x].reshape(-1,128)  
        
        
        ## ID
        # U 5 S 6
        identifier = np.array([6]*18)/6
        ID = np.array([identifier[idx]]*(128)).reshape(-1,1)
        #ID = np.tile(int_to_binary_array(np.array([identifier[idx]])), (128*128, 1))
        angles=np.array(128*[angle_rad]).reshape(-1,1)
        
        scale_pos = np.array(128).astype(float)
        scale_data = np.array(255).astype(float)
                
        
        input_data = np.concatenate([ele_x/scale_pos,ele_y/scale_pos,ele_z/scale_pos,depth/scale_pos,angles,ID], axis=1)
        input_data = pos_to_freq(input_data)
        output_data = np.concatenate([us_pixel/scale_data,bscan_pixel/scale_data], axis=1)
        
        return (input_data, output_data)

    
