import torch
import os
import torch.nn as nn
from PIL import Image
import numpy as np
import zipfile
from utils import Loss_PSNR, AverageMeter
from pytorch_ssim import SSIM
from fw_sat_arch import FW_SAT

def zip_files(files_to_zip, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_to_zip in files_to_zip:
            if os.path.isfile(file_to_zip):
                zipf.write(file_to_zip, os.path.basename(file_to_zip))


model = FW_SAT().cuda()


load_dic = torch.load('/media/max/a/2024_CVPRW_SR/FW_SAT/net_best_8.pth')



# model = nn.DataParallel(model)

model.load_state_dict(load_dic['state_dict'])


rgb_path = '../data/visible/test/guided_x8/'
lr_thermal_path = '../data/thermal/test/guided_x8/LR_x8/'



save_path = '../data/test_baseline_tcfs/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

images = os.listdir(lr_thermal_path)


for (j,i)  in enumerate(images):

    rgb = Image.open(os.path.join(rgb_path,i.replace('_th','_vis')))
    lr_thermal = Image.open(os.path.join(lr_thermal_path,i))

    rgb_image = np.array(rgb)/255.0
    lr_thermal_image = np.array(lr_thermal)/255.0

    rgb_image = np.transpose(rgb_image, (2, 0, 1))
    lr_thermal_image = np.expand_dims(lr_thermal_image[:,:,0], axis=0)

    rgb_image = np.expand_dims(rgb_image, axis=0)
    lr_thermal_image = np.expand_dims(lr_thermal_image, axis=0)

    lr_thermal_image = torch.from_numpy(lr_thermal_image).float()
    rgb_image = torch.from_numpy(rgb_image).float()
    

    with torch.no_grad():
        rgb_image = rgb_image.cuda()
        lr_thermal_image = lr_thermal_image.cuda()
        output = model(rgb_image, lr_thermal_image)
 
    output = output.cpu().numpy()
    output = np.squeeze(output)
    output = np.squeeze(output)
    output = output*255.0
    output = output.astype(np.uint8)
    output = Image.fromarray(output)
    output.save(os.path.join(save_path,i))
# zip_files(save_path, '../data/evaluation.zip')