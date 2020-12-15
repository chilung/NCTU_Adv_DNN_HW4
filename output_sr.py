
from os import listdir
from os.path import isfile, join
import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont
import datetime

testing_path = './testing_lr_images'
output_path = '/content/drive/MyDrive/NCTU/基於深度學習之視覺辨識專論/HW/HW4/output_sr_3x_2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = '/content/drive/MyDrive/NCTU/基於深度學習之視覺辨識專論/HW/HW4/checkpoint_3x_2/checkpoint_8100_srgan.pth.tar'
print('device: {}, srgan: {} '.format(device, srgan_checkpoint))

# Load models
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()

testing_files = [f for f in listdir(testing_path) if isfile(join(testing_path, f))]
testing_files.sort()
print('number of testing samples: {}'.format(len(testing_files)))
print('list of testing samples: {}'.format(testing_files))

os.makedirs(output_path, exist_ok=True)

for testing_file in testing_files:
    lr_img = Image.open(join(testing_path, testing_file), mode="r")
    lr_img = lr_img.convert('RGB')
    print('processing: {}'.format(testing_file), end = '')
    
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    
    sr_img_srgan.save(join(output_path, testing_file))
    print(' Done')
