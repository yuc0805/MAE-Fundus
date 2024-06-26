import math
import os
from typing import Any, Tuple, Union
from model import MAE #MAE_linear_probing
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import timm
import torch
from torch import distributed
import torchvision
import argparse
from datamodule import FundusDataModule
from model import *


# kubectl cp data/EAMDR physio-model-l-58f79fd5f6-vn24p:data -c physio-model-l
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Masked Autoencoder with PyTorch Lightning")
    parser.add_argument('--data_dir', type=str, default= '/Users/leo/Desktop/MAE-Fundus/data/EAMDR', 
                        help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs to use for training')
    parser.add_argument('--strategy', type=str, default='ddp', help='ddp/ddp_find_unused_parameters_true')
    parser.add_argument('--accelerator', type=str, default='gpu', help='cpu/cuda')
    parser.add_argument('--epoches', type=int, default=100, help='Number of training epoches')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.gpus))

    print('Number of available GPUs:', torch.cuda.device_count())
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))

    transform = MAETransform()
    data_module = FundusDataModule(args.data_dir,transform=transform)
    model = MAE()
    
    # dataloader = data_module.train_dataloader()

    # Train with DDP on multiple gpus. Distributed sampling is also enabled with
    # replace_sampler_ddp=True.
    
    trainer = pl.Trainer(
        max_epochs=args.epoches,
        devices="auto",
        accelerator=args.accelerator,
        strategy=args.strategy,
        use_distributed_sampler=True,  
    )
    trainer.fit(model=model, train_dataloaders=data_module)


    # python -m train_main --accelerator cpu --strategy ddp_find_unused_parameters_true
    # tensorboard --logdir /Users/leo/Desktop/MAE-Fundus/lightning_logs/version_11
    # tensorboard --logdir=lightning_logs/version_11 --host localhost --port 8888
    # python -m train_main --accelerator gpu --data_dir data/EAMDR 