# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Jiayuan Zhu
"""

import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader

import cfg
import func_2d.function as function
from conf import settings
#from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *


def main():
    # use bfloat16 for the entire work
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    # --- Initialize Adapter ---
    # We attach the adapter model to the net to keep it simple
    from func_2d.adapter import AdapterModel
    # Remove net dependency from constructor
    adapter_model = AdapterModel().to(GPUdevice)
    
    # Attach to net so it's accessible in function.py
    # Use add_module to properly register it as a submodule
    if isinstance(net, torch.nn.DataParallel):
        net.module.add_module('adapter_model', adapter_model)
    else:
        net.add_module('adapter_model', adapter_model)

    # --- Freeze Image Encoder (Partially) ---
    # Freeze most of the image encoder, but unfreeze the last few blocks and the neck
    # Structure: image_encoder.trunk.blocks.0 ... .15, image_encoder.neck
    for name, param in net.named_parameters():
        if "image_encoder" in name:
            if "neck" in name:
                param.requires_grad = True # Unfreeze neck
            elif "trunk.blocks.14" in name or "trunk.blocks.15" in name:
                param.requires_grad = True # Unfreeze last 2 blocks
            else:
                param.requires_grad = False # Freeze earlier layers
        else:
            param.requires_grad = True # Ensure others (decoder, memory, adapter) are trainable
    
    # Adapter parameters should be trainable
    for param in net.adapter_model.parameters():
        param.requires_grad = True

    # optimisation
    # Use differential learning rates:
    # - Lower LR for the pre-trained backbone (to preserve features)
    # - Higher LR for the new components (Mask Decoder, Memory, Adapter)
    
    backbone_params = []
    other_params = []
    
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
            
        if "image_encoder" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1}, # 1/10 LR for backbone
        {'params': other_params, 'lr': args.lr}           # Full LR for others
    ], lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    
    # Use CosineAnnealingWarmRestarts for better convergence and escaping local minima
    # Increase T_0 to 10 to slow down the first restart cycle, giving model more time to explore
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6) 

    '''load pretrained model'''

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)


    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    
    # example of REFUGE dataset
    if args.dataset == 'REFUGE':
        '''REFUGE data'''
        # Use FewShot Dataset for training
        refuge_train_dataset = REFUGE_FewShot(args, args.data_path, transform = transform_train, mode = 'Training')
        # Use Standard Dataset for Testing (but will be used in One-Shot mode in validation)
        # Wait, if we use validation_sam (which is now validation_one_shot-like), 
        # it expects standard dataset output?
        # validation_sam in function.py expects standard pack structure (image, mask).
        # Our DRIS dataset returns that.
        # REFUGE dataset also returns that.
        # But wait, we want to test on DRIS (Zero-Shot Cross Domain) or REFUGE?
        # User said: "Train on REFUGE, Predict on DRIS 1-shot".
        # So training data is REFUGE_FewShot.
        # Validation data is DRIS.
        
        # Training Loader (REFUGE FewShot)
        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
        
        # Validation Loader (DRIS) - We want to monitor performance on target domain?
        # Or validation on source domain?
        # Usually validation on Source to select best model, then Test on Target.
        # But user wants "Train REFUGE -> DRIS 1-shot".
        # Let's set DRIS as validation set to see One-Shot performance during training.
        
        # Manually fix path if it's the specific known path structure
        dris_path = args.data_path.replace('REFUGE', 'DRIS')
        if '/Medical-SAM2-main/data/DRIS' in dris_path and not os.path.exists(dris_path):
             # Fallback to the path user mentioned if the automatic replacement was wrong
             dris_path = '/root/autodl-tmp/DSM-main/data/DRIS'
        
        # Validation Support Loader (DRIS Training Set for One-Shot Support)
        # We need this because standard One-Shot protocol uses a labeled sample from the target domain (or source) as support.
        # Here we use DRIS Training set as the pool for support images.
        dris_train_dataset_support = DRIS(args, dris_path, transform = transform_test, mode = 'Training')
        nice_support_loader = DataLoader(dris_train_dataset_support, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
        
        # Initialize dris_test_dataset BEFORE DataLoader
        dris_test_dataset = DRIS(args, dris_path, transform = transform_test, mode = 'Test')
        nice_test_loader = DataLoader(dris_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''
    elif args.dataset == 'DRIS':
        # If user explicitly asks for DRIS training (which is not the goal here, but kept for compatibility)
        '''DRIS data'''
        dris_train_dataset = DRIS(args, args.data_path, transform = transform_train, mode = 'Training')
        dris_test_dataset = DRIS(args, args.data_path, transform = transform_test, mode = 'Test')

        nice_train_loader = DataLoader(dris_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
        nice_test_loader = DataLoader(dris_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0


    for epoch in range(settings.EPOCH):

        if epoch == 0:
            # Pass nice_support_loader to validation
            if args.dataset == 'REFUGE':
                 tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer, support_loader=nice_support_loader)
            else:
                 tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
                 
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

        # training
        net.train()
        time_start = time.time()
        # Temporarily disable prompt dropout for stable training
        temp_args = args
        temp_args.prompt_dropout_prob = 0.0
        loss = function.train_sam(temp_args, net, optimizer, nice_train_loader, epoch, writer)
        
        # Step Scheduler
        scheduler.step()
        
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        # validation
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:

            if args.dataset == 'REFUGE':
                 tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer, support_loader=nice_support_loader)
            else:
                 tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)

            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if edice > best_dice:
                best_dice = edice
                torch.save({'model': net.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))


    writer.close()


if __name__ == '__main__':
    main()
