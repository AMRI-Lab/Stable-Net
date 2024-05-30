import numpy as np
import os
from numpy import fft
import torch
from skimage.metrics import structural_similarity as compute_ssim
from scipy.io import loadmat
import utils
import model
import time
import scipy.io as sio
from skimage.io import imsave
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DEVICE = torch.device('cuda:{}'.format(str(0) if torch.cuda.is_available() else 'cpu'))


## Seed setting
seed_value = 3407 
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

## Load fully sampled k-space data
fpath ='./data/brain.mat'
f = sio.loadmat(fpath)
data_cpl = f['Y'][:]
Nchl,Nrd,Npe = data_cpl.shape

## Load unersampling mask
maskpath = './mask/mask.mat'
mask = sio.loadmat(maskpath)
SamMask  = mask['mask'][:]

## Set the path of saving results
outpath = './results'
if not os.path.exists(outpath):
    os.mkdir(outpath)

## Parameter settings
w0 = 20       
lamda = 0.5   
fn=lambda x: utils.normalize01(np.abs(x))

## Calculate the sum-of-square ground truth image
img_all = fft.fftshift(fft.ifft2(fft.fftshift(data_cpl,axes=(-1,-2)),axes=(-1,-2)),axes=(-1,-2))
gt = np.sqrt(np.sum(np.abs(img_all)**2,0))

## Perform undersampling k-space
tstKsp = data_cpl.transpose(1,2,0)
tstDsKsp = tstKsp*SamMask

## Normalize the undersampled k-space
zf_coil_img = fft.ifft2(tstDsKsp,axes=(0,1))
NormFactor = np.max(np.sqrt(np.sum(np.abs(zf_coil_img)**2,axis=2)))
tstDsKsp = tstDsKsp/NormFactor

## Reconstruct the MR image
time_all_start = time.time()
pre_img, pre_img_dc, pre_ksp = model.Recon(tstDsKsp,SamMask,DEVICE,w0=w0,TV_weight=lamda,PolyOrder=15,MaxIter=1000,LrImg = 1e-4)
  
normOrg = fn(gt)
normRec = fn(pre_img_dc) 

# Note that the psnr and ssim here are computed on the whole image including the background region.
# This is different from the results reported in the paper. 
psnrRec = utils.myPSNR(normOrg,normRec)
ssimRec = compute_ssim(normRec,normOrg,data_range=1,gaussian_weights=True)
print('{1:.4f} {0:.3f}'.format(psnrRec,ssimRec))

## Save the results
imsave(outpath + '/' + 'gt.png',normOrg)
imsave(outpath + '/' + 'recon.png',normRec)