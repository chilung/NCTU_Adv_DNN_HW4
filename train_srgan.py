import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *
from IPython.display import clear_output

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', help='the path to the root directory of model checkpoint, such as ./checkpoint')
parser.add_argument('-c', '--checkpoint', help='the path to the model checkpoint where the resume training from, such as checkpoint_srgan_8100.pth.tar')
parser.add_argument('-s', '--srresnet', help='the filepath of the trained SRResNet checkpoint used for initialization, such as checkpoint_srresnet.pth.tar')
parser.add_argument('-v', '--vggloss', action='store_true', default=False, help='set True to apply vgg on the loss function')
parser.add_argument('-e', '--epochs', type=int, default=0, help='number of epochs')
parser.add_argument('-o', '--olr', type=float, help='overwrite learning rate')

args = parser.parse_args()
print('root: {}'.format(args.root))
print('chechpoint: {}'.format(args.checkpoint))
print('loss: {}'.format(args.vggloss))
print('epochs: {}'.format(args.epochs))

# Data parameters
data_folder = './'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
# scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
scaling_factor = 3  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Generator parameters
large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks_g = 28  # number of residual blocks
srresnet_checkpoint = args.srresnet  # filepath of the trained SRResNet checkpoint used for initialization

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 14  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

checkpoint_path = args.root if not args.root == None else './'
os.makedirs(checkpoint_path, exist_ok=True)

checkpoint = args.checkpoint  # path to model (SRGAN) checkpoint, None if not specified

# Learning parameters
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 200000  # number of training iterations
# epochs = args.epoch if not args.epoch==None else 2000
workers = 4  # number of workers for loading data in the DataLoader
vgg_loss_enable = True if args.vggloss == True else False
vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
# lr = 1e-2  # learning rate
grad_clip = None  # clip if gradients are exploding

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True

# lr_base  = {15000: 0.1, 25000:0.1, 30000: 0.1, 35000: 0.1}
# def lr_table(epoch):
#     if epoch in lr_base:
#         return lr_base[epoch]
#     return 1


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, srresnet_checkpoint, vgg_loss_enable
    print(vgg_loss_enable, args.vggloss)

    save_model = False
    
    # Initialize model or load checkpoint
    if checkpoint is None:
        # Generator
        min_p_loss = 1e10
        
        generator = Generator(large_kernel_size=large_kernel_size_g,
                              small_kernel_size=small_kernel_size_g,
                              n_channels=n_channels_g,
                              n_blocks=n_blocks_g,
                              scaling_factor=scaling_factor)
        best_generator = generator

        # Initialize generator network with pretrained SRResNet
        generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)

        # Initialize generator's optimizer
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=lr)

        # Discriminator
        discriminator = Discriminator(kernel_size=kernel_size_d,
                                      n_channels=n_channels_d,
                                      n_blocks=n_blocks_d,
                                      fc_size=fc_size_d)
        best_discriminator = discriminator

        # Initialize discriminator's optimizer
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=lr)

    else:
        checkpoint = os.path.join(args.root, args.checkpoint)
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        generator = checkpoint['generator']
        best_generator = generator
        discriminator = checkpoint['discriminator']
        best_discriminator = discriminator
        optimizer_g = checkpoint['optimizer_g']
        optimizer_d = checkpoint['optimizer_d']
        min_p_loss = checkpoint['min_p_loss']
        print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))

    if args.olr != None:
        overwrite_learning_rate(optimizer_g, args.olr)
        overwrite_learning_rate(optimizer_d, args.olr)

    if vgg_loss_enable:
        # Truncated VGG19 network to be used in the loss calculation
        print('vggloss enable')
        truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
        truncated_vgg19.eval()
    else:
        truncated_vgg19 = None

    # Loss functions
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # Move to default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    if vgg_loss_enable:
        truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)
    
    # Total number of epochs to train for
    print('iterations: {}'.format(iterations))
    if args.epochs == 0:
        epochs = int(iterations // len(train_loader) + 1)
    else:
        epochs = args.epochs
    print('length of train_loader: {}'.format(len(train_loader)))
    print('epochs = {}'.format(epochs))

    # Epochs
    print('epochs: {}'.format(epochs))
    for epoch in range(start_epoch, epochs):

        # At the halfway point, reduce learning rate to a tenth
        if epoch == int((iterations / 2) // len(train_loader) + 1):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)
        
        # adjust_rate = lr_table(epoch)
        # if not adjust_rate == 1:
        #     print('adjust learning rate by {}'.format(adjust_rate))
        #     adjust_learning_rate(optimizer_g, adjust_rate)
        #     adjust_learning_rate(optimizer_d, adjust_rate)

        # One epoch's training
        p_loss = train(train_loader=train_loader,
              generator=generator,
              discriminator=discriminator,
              truncated_vgg19=truncated_vgg19,
              content_loss_criterion=content_loss_criterion,
              adversarial_loss_criterion=adversarial_loss_criterion,
              optimizer_g=optimizer_g,
              optimizer_d=optimizer_d,
              epoch=epoch)

        # Save checkpoint
        if p_loss < min_p_loss:
            best_generator = generator
            best_discriminator = discriminator
            min_p_loss = p_loss
            save_model = True
            
        if save_model == True:
            print('save model epoch {} min_p_loss: {}'.format(epoch, min_p_loss))
            torch.save({'epoch': epoch,
                'generator': best_generator,
                'discriminator': best_discriminator,
                'optimizer_g': optimizer_g,
                'optimizer_d': optimizer_d,
                'min_p_loss': min_p_loss},
                os.path.join(checkpoint_path, 'checkpoint_srgan_{}.pth.tar'.format(epoch)))
            save_model = False


def train(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch):
    """
    One epoch's training.

    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)
    :param adversarial_loss_criterion: adversarial loss function (Binary Cross-Entropy loss)
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    """
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator

    start = time.time()
        
    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), imagenet-normed

        # GENERATOR UPDATE

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
        sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm')  # (N, 3, 96, 96), imagenet-normed

        if truncated_vgg19 != None:
            # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # detached because they're constant, targets
        else:
            sr_imgs_in_vgg_space = sr_imgs
            hr_imgs_in_vgg_space = hr_imgs  

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        if truncated_vgg19 != None:
            content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        else:
            content_loss = content_loss_criterion(sr_imgs, hr_imgs) # Use original image
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()

        # Keep track of loss
        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
        # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
        # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
        # See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

        # Keep track of batch times
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]--'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})--'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})--'
                  'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})--'
                  'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})--'
                  'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})--'.format(epoch,
                       i,
                       len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss_c=losses_c,
                       loss_a=losses_a,
                       loss_d=losses_d
                       ), end = '')
    for p in optimizer_g.param_groups:
        print('learning rate: {}'.format(p['lr']))

    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated  # free some memory since their histories may be stored

    return perceptual_loss

if __name__ == '__main__':
    main()
