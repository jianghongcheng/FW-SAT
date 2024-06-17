import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class ValidDataset(Dataset):
    def __init__(self, upscale=8):
        """
        Dataset for validation images.
        
        Parameters:
        upscale (int): Upscaling factor (8 or 16) for the low-resolution thermal images.
        """
        self.HR_vis_dir = '../data/visible/val/'
        self.HR_thermal_dir = '../data/thermal/val/GT/'
        self.LR_thermal_dir = f'../data/thermal/val/LR_x{upscale}/'
        
        self.keys = sorted(os.listdir(self.HR_thermal_dir))
    
    def __getitem__(self, index):
        """
        Get a data sample for validation.
        
        Parameters:
        index (int): Index of the sample.
        
        Returns:
        tuple: Low-resolution thermal, high-resolution visible, and high-resolution thermal images as tensors.
        """
        key = self.keys[index]
        
        HR_thermal = Image.open(os.path.join(self.HR_thermal_dir, key))
        HR_vis = Image.open(os.path.join(self.HR_vis_dir, key.replace('_th.bmp', '_vis.bmp')))
        LR_thermal = Image.open(os.path.join(self.LR_thermal_dir, key))
        
        HR_vis = np.transpose(np.array(HR_vis) / 255.0, (2, 0, 1))
        HR_thermal = np.expand_dims(np.array(HR_thermal)[:, :, 0] / 255.0, axis=0)
        LR_thermal = np.expand_dims(np.array(LR_thermal)[:, :, 0] / 255.0, axis=0)
        
        return torch.tensor(LR_thermal, dtype=torch.float32), \
               torch.tensor(HR_vis, dtype=torch.float32), \
               torch.tensor(HR_thermal, dtype=torch.float32)
    
    def __len__(self):
        return len(self.keys)

class RandomTrainDataset(Dataset):
    def __init__(self, crop_size, augment=True, dbg=False, upscale=8):
        """
        Dataset for training with random crops and optional augmentations.
        
        Parameters:
        crop_size (int): Size of the crops to extract from the images.
        augment (bool): Whether to apply data augmentation.
        dbg (bool): Debug mode flag.
        upscale (int): Upscaling factor (8 or 16) for the low-resolution thermal images.
        """
        self.HR_vis_dir = '../data/visible/train/'
        self.HR_thermal_dir = '../data/thermal/train/GT/'
        self.LR_thermal_dir = f'../data/thermal/train/LR_x{upscale}/'
        self.upscale = upscale
        self.augment = augment
        self.crop_size = crop_size
        self.dbg = dbg
        
        self.keys = sorted(os.listdir(self.HR_thermal_dir))
        
        self.hr_images = []
        self.rgb_images = []
        self.lr_images = []
        
        for key in self.keys:
            hr = np.array(Image.open(os.path.join(self.HR_thermal_dir, key))) / 255.0
            rgb = np.array(Image.open(os.path.join(self.HR_vis_dir, key.replace('_th.bmp', '_vis.bmp')))) / 255.0
            lr = np.array(Image.open(os.path.join(self.LR_thermal_dir, key))) / 255.0
            
            self.hr_images.append(hr)
            self.rgb_images.append(rgb)
            self.lr_images.append(lr)
    
    def augment_image(self, img, rotTimes, vFlip, hFlip):
        """
        Apply random augmentation to an image.
        
        Parameters:
        img (np.array): Image to augment.
        rotTimes (int): Number of 90-degree rotations.
        vFlip (bool): Whether to flip vertically.
        hFlip (bool): Whether to flip horizontally.
        
        Returns:
        np.array: Augmented image.
        """
        for _ in range(rotTimes):
            img = np.rot90(img, axes=(1, 2))
        if vFlip:
            img = img[:, :, ::-1]
        if hFlip:
            img = img[:, ::-1, :]
        return img
    
    def __getitem__(self, idx):
        """
        Get a data sample for training with random crop and augmentation.
        
        Parameters:
        idx (int): Index of the sample.
        
        Returns:
        tuple: Low-resolution thermal, high-resolution visible, and high-resolution thermal images as tensors.
        """
        rgb = self.rgb_images[idx]
        hr = self.hr_images[idx]
        lr = self.lr_images[idx]
        
        h, w, _ = lr.shape
        xx = random.randint(0, h - self.crop_size)
        yy = random.randint(0, w - self.crop_size)
        
        crop_rgb = rgb[xx*self.upscale:xx*self.upscale+self.crop_size*self.upscale, yy*self.upscale:yy*self.upscale+self.crop_size*self.upscale, :]
        crop_hr = hr[xx*self.upscale:xx*self.upscale+self.crop_size*self.upscale, yy*self.upscale:yy*self.upscale+self.crop_size*self.upscale, 0]
        crop_lr = lr[xx:xx+self.crop_size, yy:yy+self.crop_size, 0]
        
        crop_rgb = np.transpose(crop_rgb, (2, 0, 1))
        crop_hr = np.expand_dims(crop_hr, axis=0)
        crop_lr = np.expand_dims(crop_lr, axis=0)
        
        if self.augment:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            
            crop_lr = self.augment_image(crop_lr, rotTimes, vFlip, hFlip)
            crop_rgb = self.augment_image(crop_rgb, rotTimes, vFlip, hFlip)
            crop_hr = self.augment_image(crop_hr, rotTimes, vFlip, hFlip)
        
        crop_lr = np.ascontiguousarray(crop_lr, dtype=np.float32)
        crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.float32)
        crop_hr = np.ascontiguousarray(crop_hr, dtype=np.float32)
        
        return torch.tensor(crop_lr), torch.tensor(crop_rgb), torch.tensor(crop_hr)
    
    def __len__(self):
        return len(self.keys)
