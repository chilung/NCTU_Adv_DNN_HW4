import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

print('cuda: {}, srgan: {} srresnet: {}'.format(device, srgan_checkpoint, srresnet_checkpoint))

# Load models
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()
