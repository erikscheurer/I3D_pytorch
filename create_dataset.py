import random

import cv2
import numpy as np
import torch
import os
import torch.nn as nn
import argparse

from functools import lru_cache
from torchvision.transforms import Resize, InterpolationMode


_VIDEO_EXT = ['.avi', '.mp4', '.mov']
_IMAGE_EXT = ['.jpg', '.png']
_IMAGE_SIZE = 224

## FlowUnderAttack load model
import sys
sys.path.append(f'./FlowUnderAttack/')
sys.path.append(f'./FlowUnderAttack/flow_library')
from flow_library.flow_IO import writeFlowFile
import helper_functions.ownutilities as ownutilities
from helper_functions.ownutilities import show_images
net = 'RAFT'#'FlowNetC'#
custom_weight_path = './FlowUnderAttack/models/_pretrained_weights/raft-sintel.pth'#FlowNet2-C_checkpoint.pth.tar'#

model_takes_unit_input = ownutilities.model_takes_unit_input(net)
model, path_weights = ownutilities.import_and_load(net, custom_weight_path=custom_weight_path, make_unit_input=not model_takes_unit_input, variable_change=False, make_scaled_input_model=True,device='cuda')
model.eval()
for p in model.parameters():
    p.requires_grad = False

### Iterate over rgb dataset, load 2 images, compute flow, save flow in the same folder structure. Two images are combined to one flow file.


def upsample_flow(flow, h, w):
    # from github.com/f-ilic/afd
    # usefull function to bring the flow to h,w shape
    # so that we can warp effectively an image of that size with it
    batched = len(flow.shape) == 3
    if batched:
        flow = flow[None, ...] # add batch dimension
    batch, h_new, w_new, _ = flow.shape
    flow_correction = torch.Tensor((h / h_new, w / w_new))
    f = (flow * flow_correction).permute(0, 3, 1, 2)

    f = (Resize((h, w), interpolation=InterpolationMode.BICUBIC)(f)).permute(0, 2, 3, 1)
    if batched:
        f = f[0, ...]
    return f


@lru_cache(maxsize=1)
def get_patch_and_defense(
    patch = '',
    defense = '',
    ):
    if patch != '':
        from helper_functions.patch_adversary import PatchAdversary
        P = PatchAdversary(patch, size = 100, angle = [-10,10], scale = [.95,1.05]) # Change of variable is always false for evaluation
        P = P.cuda().requires_grad_(False)
    else:
        P = lambda x,y: (x,y,1)

    if defense != '' or defense.lower() != 'none':
        from helper_functions.defenses import ILP,LGS
        if defense == 'ILP':
            print(f"ILP defense with k = {16}, o = {8}, t = {.15}, s = {15}, r = {5}")
            D = ILP(16, 8, .15, 15, 5, "forward")
        elif defense == 'LGS':
            print(f"LGS defense with k = {16}, o = {8}, t = {.15}, s = {15}")
            D = LGS(16, 8, .15, 15, "forward")
        else:
            print("No defense")
            D = lambda x,y: (x,y)
    return P,D

def read_image(image_path:str):
    img = np.array(cv2.imread(image_path))
    w, h, c = img.shape
    if w < 226 or h < 226:
        d = 226. - min(w, h)
        sc = 1 + d / min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    return img

def compute_and_save_flow(images, opts, save_path:str, upsample_factor:float=1):
    P,D = get_patch_and_defense(opts.patch_path, opts.defense)
    flow = []

    bins = np.linspace(-20, 20, num=256)
    for i,(img1, img2) in enumerate(zip(images[:-1:2], images[1::2])):
        img1 = read_image(img1)
        img2 = read_image(img2)
        frame1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().cuda().requires_grad_(False)
        frame2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().cuda().requires_grad_(False)

        shape = frame1.shape
        I1 = nn.Upsample(scale_factor=upsample_factor, mode='bilinear')(frame1)
        I2 = nn.Upsample(scale_factor=upsample_factor, mode='bilinear')(frame2)
        padder, [I1, I2] = ownutilities.preprocess_img(net, I1, I2)
        if not model_takes_unit_input:
            I1 = I1 / 255.
            I2 = I2 / 255.
            
        I1, I2,*_ = P(I1,I2)
        I1, I2 = D(I1,I2)
            
        curr_flow = ownutilities.compute_flow(model,"scaled_input_model",I1,I2)
        [curr_flow] = ownutilities.postprocess_flow(net, padder, curr_flow)
        # downsample flow
        curr_flow = upsample_flow(curr_flow.permute(0,2,3,1).cpu(), shape[2], shape[3])
        curr_flow = curr_flow.numpy()[0]

        # Save flow
        folder_base = os.path.basename(save_path)
        writeFlowFile(curr_flow.squeeze(), os.path.join(save_path, f'{folder_base}-{i+1:06d}.flo'))
        flow.append(curr_flow)
    return flow

def process_video(folder_path:str, goal_path:str, opts):
    # Read all images from the folder
    images = []
    for file in os.listdir(folder_path):
        if os.path.splitext(file)[1] in _IMAGE_EXT:
            images.append(os.path.join(folder_path, file))

    # Compute flow
    compute_and_save_flow(images, opts, goal_path)


def process_dataset(dataset_path:str, opts):
    for folder in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, folder)):
            goal_path = os.path.join(opts.new_dataset_path, folder)
            if not os.path.exists(goal_path):
                os.makedirs(goal_path)
            process_video(os.path.join(dataset_path, folder),goal_path, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_path', type=str, default='', help='Path to the patch')
    parser.add_argument('--defense', type=str, default='', help='Defense to use')
    parser.add_argument('--original_dataset_path', type=str, default='data/Charades_v1_rgb', help='Path to the original dataset')
    parser.add_argument('--new_dataset_path', type=str, default='data/Charades_v1_flow', help='Path to the new dataset')
    opts = parser.parse_args()


    process_dataset(opts.original_dataset_path, opts)