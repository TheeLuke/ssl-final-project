import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Import our custom modules
import utils
from models import get_vit_small_dino
from loss import DINOLoss
from dataset import load_combined_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('DINO training script', add_help=False)
    
    # Model parameters
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--out_dim', default=65536, type=int, help='Dimensionality of the DINO head output.')
    
    # Training/Optimization parameters
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="Whether or not to weight normalize the last layer of the DINO head.")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="Base EMA parameter for teacher update.")
    parser.add_argument('--use_fp16', default=True, type=utils.bool_flag, help="Whether or not to use half precision for training")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="Initial weight decay.")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="Final weight decay.")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Max gradient norm.")
    parser.add_argument('--batch_size', default=64, type=int, help='Per-GPU batch-size.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="Number of epochs to freeze last layer.")
    parser.add_argument('--lr', default=0.0005, type=float, help="Base learning rate.")
    parser.add_argument('--warmup_epochs', default=10, type=int, help="Number of warmup epochs.")
    
    # Data parameters
    parser.add_argument('--data_path_provided', default='./pretrain', type=str)
    parser.add_argument('--data_path_external', default='./external_data', type=str)
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    
    return parser

def train_dino(args):
    # 1. Prepare Output Directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. Prepare Data
    print(f"Loading data...")
    dataset = load_combined_dataset(args.data_path_provided, args.data_path_external)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    
    # 3. Build Student and Teacher Networks
    print(f"Creating model with patch size {args.patch_size}...")
    student = get_vit_small_dino(patch_size=args.patch_size, out_dim=args.out_dim)
    teacher = get_vit_small_dino(patch_size=args.patch_size, out_dim=args.out_dim)
    
    # Move to GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    student, teacher = student.to(device), teacher.to(device)

    # Teacher and student start with same weights
    teacher.load_state_dict(student.state_dict())
    
    # Turn off gradients for teacher
    for p in teacher.parameters():
        p.requires_grad = False
        
    # 4. Loss
    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        nepochs=args.epochs,
    ).to(device)

    # 5. Optimizer
    params_groups = utils.get_params_groups(student) # You might need to add this helper to utils.py or just use standard param groups
    # Standard AdamW param groups setup:
    param_groups = [
        {'params': [p for n, p in student.named_parameters() if ('bias' not in n and 'norm' not in n)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student.named_parameters() if ('bias' in n or 'norm' in n)], 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(param_groups)

    # 6. Schedulers
    # Learning rate schedule
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size / 256.),  # Linear scaling rule
        1e-6,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    
    # Weight decay schedule
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    
    # Momentum schedule (start at 0.996, go to 1.0)
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))

    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Training One Epoch
        metric_logger = utils.AverageMeter()
        student.train()
        
        # Loop over batches
        for it, (images, _) in enumerate(data_loader):
            # images is a list of tensors (crops)
            # Move all crops to GPU
            images = [im.to(device, non_blocking=True) for im in images]
            
            # Step-based scheduling
            it_schedule = it + len(data_loader) * epoch
            cur_lr = lr_schedule[it_schedule]
            cur_wd = wd_schedule[it_schedule]
            cur_mom = momentum_schedule[it_schedule]
            
            # Update optimizer learning rate and weight decay
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = cur_lr
                if i == 0: # Only apply weight decay to first group
                    param_group['weight_decay'] = cur_wd

            # Forward pass
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                # Student forward (all crops)
                student_output = student(images) 
                # Teacher forward (global crops only - first 2)
                teacher_output = teacher(images[:2])
                
                loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)

            # Backprop
            optimizer.zero_grad()
            
            # Scale loss if using FP16 (Standard PyTorch scaler recommended)
            # For simplicity here:
            loss.backward()
            
            # Clip gradients
            utils.clip_gradients(student, args.clip_grad)
            
            # Cancel gradients for last layer in first epoch
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            
            optimizer.step()

            # Teacher EMA Update
            with torch.no_grad():
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(cur_mom).add_((1 - cur_mom) * param_q.detach().data)

            # Logging
            if it % 10 == 0:
                print(f"Epoch: [{epoch}/{args.epochs}] Step: [{it}/{len(data_loader)}] "
                      f"Loss: {loss.item():.4f} LR: {cur_lr:.6f}")

        # Save Checkpoint
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        utils.save_checkpoint(os.path.join(args.output_dir, f'checkpoint.pth'), save_dict)
        
        if epoch % 10 == 0:
             utils.save_checkpoint(os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'), save_dict)

    total_time = time.time() - start_time
    print(f"Training time: {str(datetime.timedelta(seconds=int(total_time)))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    train_dino(args)