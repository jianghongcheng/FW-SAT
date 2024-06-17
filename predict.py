import os
import torch
import numpy as np
from PIL import Image
import zipfile
from fw_sat_arch import FW_SAT
from utils import Loss_PSNR, AverageMeter
from pytorch_ssim import SSIM

def zip_files(directory_to_zip, output_zip_path):
    """
    Zips the contents of a directory.

    Args:
        directory_to_zip (str): Path to the directory containing files to zip.
        output_zip_path (str): Path to the output zip file.
    """
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_to_zip):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, directory_to_zip))

def load_model(checkpoint_path):
    """
    Loads the model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = FW_SAT().cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def preprocess_image(image_path, expand_dims=True):
    """
    Preprocesses an image for model input.

    Args:
        image_path (str): Path to the image.
        expand_dims (bool): Whether to expand dimensions for batch size.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path)
    image = np.array(image) / 255.0
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
    else:
        image = np.expand_dims(image, axis=0)
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image).float()

def save_image(output, save_path, image_name):
    """
    Saves the output image to the specified path.

    Args:
        output (torch.Tensor): Output image tensor.
        save_path (str): Directory to save the image.
        image_name (str): Name of the image file.
    """
    output = output.cpu().numpy().squeeze() * 255.0
    output = output.astype(np.uint8)
    output_image = Image.fromarray(output)
    output_image.save(os.path.join(save_path, image_name))

def process_images(model, rgb_dir, lr_thermal_dir, save_dir):
    """
    Processes images with the model and saves the output.

    Args:
        model (torch.nn.Module): The loaded model.
        rgb_dir (str): Directory containing RGB images.
        lr_thermal_dir (str): Directory containing low-resolution thermal images.
        save_dir (str): Directory to save the processed images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_files = os.listdir(lr_thermal_dir)
    for image_name in image_files:
        rgb_image_path = os.path.join(rgb_dir, image_name.replace('_th', '_vis'))
        lr_thermal_image_path = os.path.join(lr_thermal_dir, image_name)

        rgb_image = preprocess_image(rgb_image_path)
        lr_thermal_image = preprocess_image(lr_thermal_image_path, expand_dims=False)
        lr_thermal_image = lr_thermal_image.unsqueeze(0)

        with torch.no_grad():
            rgb_image = rgb_image.cuda()
            lr_thermal_image = lr_thermal_image.cuda()
            output = model(rgb_image, lr_thermal_image)

        save_image(output, save_dir, image_name)

def main():
    checkpoint_path = '/media/max/a/2024_CVPRW_SR/FW_SAT/net_best_8.pth'
    rgb_dir = '../data/visible/test/guided_x8/'
    lr_thermal_dir = '../data/thermal/test/guided_x8/LR_x8/'
    save_dir = '../data/test/'

    model = load_model(checkpoint_path)
    process_images(model, rgb_dir, lr_thermal_dir, save_dir)
    # Uncomment the line below to zip the output files
    # zip_files(save_dir, '../data/evaluation.zip')

if __name__ == '__main__':
    main()
