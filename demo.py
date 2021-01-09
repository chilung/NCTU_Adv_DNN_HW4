# import common utilities
import os
 
# import google colab utilities
from google.colab import drive
from google.colab.patches import cv2_imshow
from google_drive_downloader import GoogleDriveDownloader as gdd
 
# import pytorch utilities
import torch
import torchvision

gdd.download_file_from_google_drive(file_id='1B7QlA1HPH7zMENuYXDxlbzgNLbJqS1zJ',
                dest_path='./testing_lr_images.zip',
                unzip=True)
gdd.download_file_from_google_drive(file_id='1bIowNL2X6zqfnmfcxqX3oPh2GIR9lxnr',
                dest_path='./training_hr_images.zip',
                unzip=True)
                
from os import listdir
from os.path import isfile, join
 
training_path = './training_hr_images'
trainingfiles = [join(training_path, f) for f in listdir(training_path) if isfile(join(training_path, f))]
print('number of training samples: {}'.format(len(trainingfiles)))
# print('list of training samples: {}'.format(trainingfiles))
 
import json
 
with open('./train_images.json', 'w') as f:
    json.dump(trainingfiles, f)

# python train_srresnet.py --help
# python train_srresnet.py -r '/content/drive/MyDrive/NCTU/基於深度學習之視覺辨識專論/HW/HW4/checkpoint_3x_3' -c 'checkpoint_srresnet_52000.pth.tar'

# python train_srgan.py --help
python train_srgan.py -r './checkpoint_3x_vgg_g28l_r14l_3'
# python train_srgan.py -o 1e-7 -v -r '/content/drive/MyDrive/NCTU/基於深度學習之視覺辨識專論/HW/HW4/checkpoint_3x_vgg_longrun' -e 60000 -c 'checkpoint_srgan_44100.pth.tar' -s '/content/drive/MyDrive/NCTU/基於深度學習之視覺辨識專論/HW/HW4/checkpoint_3x_3/checkpoint_srresnet_52630.pth.tar'
 
# python output_sr.py -o '/content/drive/MyDrive/NCTU/基於深度學習之視覺辨識專論/HW/HW4/output_3x_mse_longrun' -g '/content/drive/MyDrive/NCTU/基於深度學習之視覺辨識專論/HW/HW4/checkpoint_3x_vgg_longrun/checkpoint_srgan_27000.pth.tar'
