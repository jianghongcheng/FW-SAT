from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import torch
import random
import cv2

class ValidDataset(Dataset):
    def __init__(self, upscale = 8):

        self.HR_vis = os.path.join('../data/visible/val/')
        self.HR_thermal = os.path.join('../data/thermal/val/GT/')

        if upscale == 8:
            self.LR_thermal = os.path.join('../data/thermal/val/LR_x8/')

        else :
            self.LR_thermal = os.path.join('../data/thermal/val/LR_x16/')
        data_names = os.listdir(self.HR_thermal)
        self.keys = data_names
        self.keys.sort()
    
    def __getitem__(self, index):

        key = self.keys[index]
    
        HR_thermal = Image.open(os.path.join(self.HR_thermal, key))
        HR_vis = Image.open(os.path.join(self.HR_vis, key.replace('_th.bmp','_vis.bmp')))
        LR_thermal = Image.open(os.path.join(self.LR_thermal, key))
        
        HR_vis = np.array(HR_vis)/255.0
        HR_vis = np.transpose(HR_vis, (2, 0, 1))

        HR_thermal = np.array(HR_thermal)/255.0
        HR_thermal = np.expand_dims(HR_thermal[:,:,0], axis=0)

        LR_thermal = np.array(LR_thermal)/255.0

        LR_thermal = np.expand_dims(LR_thermal[:,:,0], axis=0)
        LR_thermal = torch.from_numpy(LR_thermal).float()
        HR_vis = torch.from_numpy(HR_vis).float()
        HR_thermal = torch.from_numpy(HR_thermal).float()

        return LR_thermal, HR_vis, HR_thermal    
    

    def __len__(self):
        return len(self.keys)


class RandomTrainDataset(Dataset):
    def __init__(self, crop_size,augment=True, dbg = False, upscale = 8):


        self.HR_vis = os.path.join('../data/visible/train/')
        self.HR_thermal = os.path.join('../data/thermal/train/GT/')

        self.upscale = upscale  
        if upscale == 8:
            self.LR_thermal = os.path.join('../data/thermal/train/LR_x8/')

        else:
            self.LR_thermal = os.path.join('../data/thermal/train/LR_x16/')    
        self.augment = augment
        self.hr = []
        self.rgb = []
        self.lr = []
        self.crop_size = crop_size
        self.dbg = dbg
        
        data_names = os.listdir(self.HR_thermal)

        self.keys = data_names
        self.keys.sort()
        
        for i in range(len(data_names)):
            
            
            hr = Image.open(os.path.join(self.HR_thermal, data_names[i]))
            rgb = Image.open(os.path.join(self.HR_vis, data_names[i].replace('_th.bmp','_vis.bmp')))
            lr = Image.open(os.path.join(self.LR_thermal, data_names[i]))
            hr = np.array(hr)/255.0
            rgb = np.array(rgb)/255.0
            lr = np.array(lr)/255.0
            self.hr.append(hr)
            self.rgb.append(rgb)
            self.lr.append(lr)

        self.img_num = len(self.hr)
        
    
    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img
    
    def __getitem__(self, idx):
       
        rgb = self.rgb[idx]
        hr = self.hr[idx]
        lr = self.lr[idx]
        h,w,c = lr.shape
        xx = random.randint(0, h-self.crop_size)
        yy = random.randint(0, w-self.crop_size)
        rgb = rgb[xx*self.upscale:xx*self.upscale+self.crop_size*self.upscale,yy*self.upscale:yy*self.upscale+self.crop_size*self.upscale,:]
        hr = hr[xx*self.upscale:xx*self.upscale+self.crop_size*self.upscale,yy*self.upscale:yy*self.upscale+self.crop_size*self.upscale,0]
        lr = lr[xx:xx+self.crop_size,yy:yy+self.crop_size,0]

        

        rgb = np.transpose(rgb, (2, 0, 1))
        hr = np.expand_dims(hr, axis=0)
        lr = np.expand_dims(lr, axis=0)
        # print(f"rgb {rgb.shape} lr {lr.shape} hr {hr.shape}")


        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        if self.augment:
            LR_thermal = self.arguement(lr, rotTimes, vFlip, hFlip)
            HR_vis = self.arguement(rgb, rotTimes, vFlip, hFlip)
            HR_thermal = self.arguement(hr, rotTimes, vFlip, hFlip)
        
        LR_thermal =  np.ascontiguousarray(LR_thermal).astype(np.float32)
        HR_vis =  np.ascontiguousarray(HR_vis).astype(np.float32)
        HR_thermal =  np.ascontiguousarray(HR_thermal).astype(np.float32)

        return LR_thermal, HR_vis, HR_thermal    
    

    def __len__(self):
        return self.img_num
    