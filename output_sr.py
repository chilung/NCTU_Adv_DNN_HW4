
from os import listdir
from os.path import isfile, join
import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont
import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', help='the path to the output directory of super resolution images.')
parser.add_argument('-g', '--gan', help='the full file path to the SRGAN super resolution model checkpoint, such as checkpoint_srgan.pth.tar')
parser.add_argument('-r', '--resnet', help='the full file path to the SRResNetsuper resolution model checkpoint, such as checkpoint_srresnet.pth.tar')

args = parser.parse_args()
print('output: {}'.format(args.output))
print('srgan model: {}'.format(args.gan))
print('srresnet model: {}'.format(args.resnet))

testing_path = './testing_lr_images'
output_path = args.output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
if args.gan != None:
    srgan_checkpoint = args.gan
    print('device: {}, srgan: {} '.format(device, srgan_checkpoint))

    # Load Gan models
    srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
    srgan_generator.eval()

    testing_files = [f for f in listdir(testing_path) if isfile(join(testing_path, f))]
    testing_files.sort()
    print('number of testing samples: {}'.format(len(testing_files)))
    print('list of testing samples: {}'.format(testing_files))

    os.makedirs(output_path+'/gan', exist_ok=True)

    for testing_file in testing_files:
        lr_img = Image.open(join(testing_path, testing_file), mode="r")
        lr_img = lr_img.convert('RGB')
        print('processing: {}'.format(testing_file), end = '')
    
        sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
        sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    
        sr_img_srgan.save(join(output_path+'/gan', testing_file))
        print('SRGAN Done')
    
# Load srresnet models
if args.resnet != None:
    srresnet_checkpoint = args.resnet
    print('device: {}, srresnet: {} '.format(device, srgan_checkpoint))

    srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
    srresnet.eval()

    testing_files = [f for f in listdir(testing_path) if isfile(join(testing_path, f))]
    testing_files.sort()
    print('number of testing samples: {}'.format(len(testing_files)))
    print('list of testing samples: {}'.format(testing_files))

    os.makedirs(output_path+'/resnet', exist_ok=True)

    for testing_file in testing_files:
        lr_img = Image.open(join(testing_path, testing_file), mode="r")
        lr_img = lr_img.convert('RGB')
        print('processing: {}'.format(testing_file), end = '')

        sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
        sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
        sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

        sr_img_srresnet.save(join(output_path+'/resnet', testing_file))
        print('SRResNet Done')
