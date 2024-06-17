import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from dataloader import  ValidDataset , RandomTrainDataset
import torch.nn.functional as F
# import utils
from utils import AverageMeter ,Loss_PSNR, save_checkpoint , VGGPerceptualLoss
import datetime

from torch.utils.tensorboard import SummaryWriter
from pytorch_ssim import SSIM
import yaml
from fw_sat_arch import FW_SAT
import torch
import time





with open('option.yaml','r') as config_file:
     config = yaml.safe_load(config_file)

import shutil

source_file = 'option.yaml'
destination_path = config['outf']

os.makedirs(destination_path, exist_ok=True)


# Ensure the destination directory exists

shutil.copy2(source_file, destination_path)

writer = SummaryWriter(config['outf'])
# load dataset


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

print("\nloading dataset ...")
train_data = RandomTrainDataset(crop_size = config['patch_size'], upscale= config['upscale_factor'])


val_data = ValidDataset(upscale= config['upscale_factor'])

train_loader = DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)

val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


total_iteration = 1000*config['end_epoch']
criterion_L1 = nn.L1Loss()
criterion_PSNR = Loss_PSNR()
criterion_SSIM = SSIM()
criterion_Perceptual = VGGPerceptualLoss()

model = FW_SAT().cuda()



# model = model.cuda()

# model = nn.DataParallel(model)


print('Parameters number is ', sum(param.numel() for param in model.parameters())) 
if torch.cuda.is_available():
    model = model.cuda()
    criterion_L1 = criterion_L1.cuda()
    criterion_PSNR = criterion_PSNR.cuda() 
    criterion_SSIM = criterion_SSIM.cuda() 
    criterion_Perceptual = criterion_Perceptual.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=float(config['init_lr']), betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

resume_file = config['resume_file']
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


iteration = 0

def validate(model,val_loader):
    model.eval()
    losses_psnr = AverageMeter()
    losses_ssim = AverageMeter()

    for i, (lr,rgb,hr) in enumerate(val_loader):
        lr = lr.cuda()
        hr = hr.cuda()
        rgb = rgb.cuda()
        with torch.no_grad():
            output = model(rgb, lr)
            loss_psnr = criterion_PSNR(output, hr)
            loss_ssim = criterion_SSIM(output, hr)
            losses_psnr.update(loss_psnr.data)
            losses_ssim.update(loss_ssim.data)
    return losses_psnr.avg, losses_ssim.avg

# torch.autograd.set_detect_anomaly(True)
record_psnr = 0
record_ssim = 0
 
best_psnr = 0
best_ssim = 0
prev_time = time.time()

while iteration <  total_iteration:
    model.train()
    losses = AverageMeter()
    losses_l1 = AverageMeter()
    losses_ssim = AverageMeter()
    losses_perceptual = AverageMeter()
    losses_ds_l1 = AverageMeter()
    for i, (lr,rgb,hr) in enumerate(train_loader):

        lr = lr.cuda()
        rgb = rgb.cuda()
        hr = hr.cuda()
        optimizer.zero_grad()
        output = model(rgb,lr)
        # print(f"rgb {rgb.shape} lr {lr.shape} output {output.shape}")
        l1_loss = criterion_L1(output, hr)
        l1_loss = criterion_L1(output, hr)
        ssim_loss = 1-criterion_SSIM(output, hr)
        perceptual_loss = criterion_Perceptual(output, hr)
        loss = 7* l1_loss + ssim_loss + 0.15 * perceptual_loss
        # loss = l1_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.data)
        losses_l1.update(l1_loss.data)
        losses_ssim.update(ssim_loss.data)
        # losses_ds_l1.update(l1_ds_loss.data)
        losses_perceptual.update(perceptual_loss.data)
        iteration = iteration+1


                    # Determine approximate time left
        iters_done = iteration 
        iters_left = total_iteration - iters_done
        time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
        prev_time = time.time()
        print(f'time_left {time_left}')

        if iteration % 100 == 0:
            print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f,train_losses_l1.avg=%.9f,train_losses_ssim.avg=%.9f,train_losses_perceptual=%.9f' 
                    % (iteration, total_iteration, optimizer.param_groups[0]['lr'], losses.avg,losses_l1.avg,losses_ssim.avg,losses_perceptual.avg))
        if iteration % (1000 * (16//config['batch_size'])) == 0:
            psnr, ssim = validate(model, val_loader)

            

            print('[epoch: %d/%d],psnr=%.4f,ssim=%.4f'%((iteration//1000),config['end_epoch'],psnr,ssim))

            writer.add_scalar('train_loss', losses.avg, iteration//1000)
            writer.add_scalar('train_loss_l1', losses_l1.avg, iteration//1000)
            writer.add_scalar('train_loss_ssim', losses_ssim.avg, iteration//1000)
            writer.add_scalar('train_loss_ds_l1', losses_ds_l1.avg, iteration//1000)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration//1000)
            writer.add_scalar('psnr', psnr, iteration//1000)
            writer.add_scalar('ssim', ssim, iteration//1000)
            losses = AverageMeter()
            losses_l1 = AverageMeter()
            losses_ssim = AverageMeter()
            losses_ds_l1 = AverageMeter()

            if (psnr > best_psnr or ssim > best_ssim) and iteration//1000 > 15: 

                if psnr > best_psnr:
                    best_psnr = psnr

                if ssim > best_ssim:
                    best_ssim = ssim    
                save_checkpoint(config['outf'], iteration//1000, iteration, model, optimizer)



