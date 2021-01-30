#import os
#import glob
import numpy as np
import cv2
from cv2 import cv2
from urllib.request import urlopen
#from PIL import Image

import seam_carving
from seam_carving import SeamCarver
#import matplotlib.pyplot as plt
#import multiprocessing as mp
#import random
#import boto3

import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#from skimage.transform import resize
#from sklearn.metrics import *

def get_img(im):
    ''' swap axes '''
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    return im 

def get_image_from_mask(pred_mask_img, o_img, N=400):

    seam_pred_mask = get_img(pred_mask_img)
    seam_pred_mask = np.mean(seam_pred_mask,axis=2)    
    col_sum = np.sum(seam_pred_mask, axis=0)
    idx_to_keep = col_sum.argsort()[:N]
    idx_to_keep.sort()
    orig_img = get_img(o_img)
    return orig_img[:, idx_to_keep]

def run_inference(np_image, model, resize_height=384, resize_width=496):
    #resize_height = 384
    #resize_width = 496

    #convert url to image numpy array
    #print("photo url:", photo_url)
    #resp = urlopen(photo_url)
    
    in_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    #To read from file path uncomment below line
    #in_image = cv2.imread(photo)
    im = cv2.resize(in_image, dsize=(resize_width, resize_height)).astype(np.float64)
    im = np.swapaxes(im, 1, 2)
    np_img = np.swapaxes(im, 0, 1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.from_numpy(np.expand_dims(np_img, axis=0)).to(device)
    
    pred_mask = model.forward(img_tensor.type(torch.FloatTensor)).detach().numpy()[0]
    output_image = get_image_from_mask(pred_mask, np_img)

    #convert BGR to RGB 
    return output_image[:,:,::-1]
 
