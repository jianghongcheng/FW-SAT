import torch
import os
import numpy as np
from PIL import Image
from fw_sat_arch import FW_SAT  # Assuming FW_SAT is defined in fw_sat_arch
from pytorch_ssim import SSIM  # Assuming SSIM is imported from pytorch_ssim module


class ImageProcessor:
    def __init__(self, model_path, rgb_path, lr_thermal_path, save_path):
        self.model = FW_SAT().cuda()
        self.model.load_state_dict(torch.load(model_path)['state_dict'])
        self.rgb_path = rgb_path
        self.lr_thermal_path = lr_thermal_path
        self.save_path = save_path

    def process_images(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        images = os.listdir(self.lr_thermal_path)

        for idx, image_name in enumerate(images):
            rgb, lr_thermal = self.load_images(image_name)

            output = self.predict(rgb, lr_thermal)

            self.save_output(output, image_name)

    def load_images(self, image_name):
        rgb = Image.open(os.path.join(self.rgb_path, image_name.replace('_th', '_vis')))
        lr_thermal = Image.open(os.path.join(self.lr_thermal_path, image_name))
        
        rgb_image = np.array(rgb) / 255.0
        lr_thermal_image = np.array(lr_thermal) / 255.0

        rgb_image = np.transpose(rgb_image, (2, 0, 1))
        lr_thermal_image = np.expand_dims(lr_thermal_image[:, :, 0], axis=0)

        rgb_image = np.expand_dims(rgb_image, axis=0)
        lr_thermal_image = np.expand_dims(lr_thermal_image, axis=0)

        lr_thermal_image = torch.from_numpy(lr_thermal_image).float()
        rgb_image = torch.from_numpy(rgb_image).float()

        return rgb_image, lr_thermal_image

    def predict(self, rgb_image, lr_thermal_image):
        with torch.no_grad():
            rgb_image = rgb_image.cuda()
            lr_thermal_image = lr_thermal_image.cuda()
            output = self.model(rgb_image, lr_thermal_image)

        output = output.cpu().numpy()
        output = np.squeeze(output) * 255.0
        output = output.astype(np.uint8)

        return output

    def save_output(self, output, image_name):
        output_image = Image.fromarray(output)
        output_image.save(os.path.join(self.save_path, image_name))




if __name__ == "__main__":
    model_path = '/media/max/a/2024_CVPRW_SR/FW_SAT/FW_SAT/exp/net_32epoch.pth'
    rgb_path = '../data/visible/test/guided_x8/'
    lr_thermal_path = '../data/thermal/test/guided_x8/LR_x8/'
    save_path = '../data/test_out/'

    image_processor = ImageProcessor(model_path, rgb_path, lr_thermal_path, save_path)
    image_processor.process_images()

