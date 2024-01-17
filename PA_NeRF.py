#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 09:29:33 2023

@author: whitaker
"""

import torch,os,math
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import numpy as np
from Data_loader import PAT_3D_train_dataset,PAT_3D_test_dataset,PAT_3D_validate_dataset,load_fluence
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import savemat
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

def ssim_loss(img1, img2):
    # Convert tensors to NumPy arrays and calculate SSIM
    img1 = img1.detach().cpu().numpy().squeeze()
    img2 = img2.detach().cpu().numpy().squeeze()
    ssim_val = ssim(img1, img2, data_range=img1.max() - img1.min())

    # Convert SSIM to PyTorch tensor and return it as the loss
    return torch.tensor(1 - ssim_val, dtype=torch.float32, requires_grad=True)


class GradientSimilarityLoss(nn.Module):
    def forward(self, input, target):
        dy_input = input[:,:,1:,:] - input[:,:,:-1,:]
        dx_input = input[:,:,:,1:] - input[:,:,:,:-1]
        
        dy_target = target[:,:,1:,:] - target[:,:,:-1,:]
        dx_target = target[:,:,:,1:] - target[:,:,:,:-1]
        
        gd_loss = torch.mean((dx_input - dx_target)**2) + torch.mean((dy_input - dy_target)**2)
        
        return gd_loss

# Define the NeRF model
class NeRF(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(NeRF, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # Define the MLP layers
        self.layer1 = nn.Linear(input_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3_1 = nn.Linear(hidden_dims, hidden_dims)
        self.layer4 = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer3_1(x))
        x = self.layer4(x)
        return x

# Define the loss function
def reconstruction_loss(output, output_data):

    # Compute the inverse rendering loss
    loss = torch.mean(((output_data - output) ** 2))
    # Compute the render loss
    
    
    return loss
#
def foward_loss(ua_map, pa_map):
    fluence_map = load_fluence()
    fluence_map = torch.tensor(fluence_map).reshape(-1,1,128,128).to(device)
    foward_pa = ua_map*fluence_map
    f_loss = ssim_loss(foward_pa, pa_map)
    foward_pa.detach()
    loss = torch.mean(((foward_pa - pa_map) ** 2))
    return loss

# Define the training function
def train_nerf(model, ind, optimizer, input_data, output_data,simu_folder,loss_list):
    # Compute the loss and update the model
    optimizer.zero_grad()
    output = model(input_data)
    mse_loss = reconstruction_loss(output, output_data)
    ua,bscan = output[:,:,:128].reshape(-1,1,128,128),output[:,:,128:].reshape(-1,1,128,128)
    ua_gt,bscan_gt = output_data[:,:,:128].reshape(-1,1,128,128),output_data[:,:,128:].reshape(-1,1,128,128)
    g_loss_bscan = loss_g(bscan,bscan_gt)
    g_loss_gt = loss_g(ua,ua_gt)
    f_loss = foward_loss(ua,bscan)

    (mse_loss+1e-1*(g_loss_bscan+g_loss_gt)+1e-1*f_loss).backward()
    loss_list.append([(mse_loss+g_loss_bscan+g_loss_gt).item(),mse_loss.item(),(g_loss_bscan+g_loss_gt).item()])
    #max(loss_mse,loss_render).backward()
    optimizer.step()
    if ind%1==0:
        #scheduler.step(mse_loss+(g_loss_bscan+g_loss_gt))
        print(str(ind)+'th of 3 batch, loss_mse:'+str(mse_loss.item())+' ,loss_g:'+str((g_loss_bscan+g_loss_gt).item())+' ,loss_f:'+str((f_loss).item()))
    

# Define the testing function
def test_nerf(model, input_data):
    # Generate radiance and density values for each position
    with torch.no_grad():
        output = model(input_data)

    return output



    
# Define the input data and generate the data loader
epochs,batch_size = 500,20
simu_folder = '/media/whitaker-160/bigstorage/NeRF/Data/Simu_1/'
phanS_folder = '/media/whitaker-160/bigstorage/NeRF/Data/Phantom/Phantom_S/'
phanU_folder = '/media/whitaker-160/bigstorage/NeRF/Data/Phantom/Phantom_U/'

model_PATH = '/media/whitaker-160/bigstorage/NeRF/model_saved/best_model.pth_all_loss'
save_path = '/media/whitaker-160/bigstorage/NeRF/model_saved/'

loss_g = GradientSimilarityLoss()
# Define the NeRF model and optimizer
model = NeRF(input_dims=120, hidden_dims=256, output_dims=2*128)
# model = torch.load(model_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-3,weight_decay=4e-7)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=30)
test_stage=0
# Train the NeRF model
if not test_stage:
    train_3D = PAT_3D_train_dataset(simu_folder,phanU_folder, phanS_folder)
    DL_train = DataLoader(train_3D, batch_size=batch_size, shuffle=True,num_workers=0,drop_last=False)
    loss_list = []
    iterator = tqdm(range(epochs), ncols=70)
    for epoch in iterator:
        for i, (input_data, output_data)  in enumerate(DL_train):
            input_data = input_data.float().to(device)
            output_data = output_data.float().to(device)
            train_nerf(model,i, optimizer, input_data, output_data,simu_folder,loss_list)
            
        if (epoch + 1) % 200 == 0:
            save_mode_path = os.path.join(save_path, 'epoch_' + str(epoch) + '.pth')
            #torch.save(model, save_mode_path)
            print("save model to {}".format(save_mode_path))
    plt.plot(loss_list)
# Test the NeRF
if test_stage: #0-> 0deg 17->170deg
    test_idx = 0
    angle_idx=6
    test_3D = PAT_3D_validate_dataset(phanS_folder,test_idx)
    DL_test = DataLoader(test_3D, batch_size=180,shuffle=False)
    test_data,test_gt= next(iter(DL_test))
    test_data = test_data.float().to(device)
    render_output = test_nerf(model,test_data)
    bscan_render=render_output[angle_idx,:,128:]
    bscan_render = bscan_render.cpu().detach().numpy().reshape(128,128)
    bscan_render[bscan_render<np.mean(bscan_render)]=0
    ua_render=render_output[angle_idx,:,:128]
    ua_render = ua_render.cpu().detach().numpy().reshape(128,128)
    ua_render[ua_render<np.mean(ua_render)]=0
    b_volume = test_gt.cpu().detach().numpy()
    bscan_volume = b_volume[angle_idx,:,128:].reshape(128,128)
    ua_volume = b_volume[angle_idx,:,:128].reshape(128,128)
    plt.figure(1)
    plt.imshow(ua_render)
    plt.figure(2)
    plt.imshow(ua_volume)
    plt.figure(3)
    plt.imshow(bscan_render)
    plt.figure(4)
    plt.imshow(bscan_volume)
    
    PAT = render_output[:,:,128:].cpu().detach().numpy()
    PE = render_output[:,:,:128].cpu().detach().numpy()
    PAT_gt = b_volume[:,:,128:]
    PE_gt = b_volume[:,:,:128]
    mdic = {'PAT_output':PAT}
    savemat("Render_output/paper/phanS_PAT_2.mat", mdic)
    mdic = {'PE_output':PE}
    savemat("Render_output/paper/phanS_PE_2.mat", mdic)
    mdic = {'PAT_gt':PAT_gt}
    savemat("Render_output/paper/PAT_gt.mat", mdic)
    mdic = {'PE_gt':PE_gt}
    savemat("Render_output/paper/PE_gt.mat", mdic)