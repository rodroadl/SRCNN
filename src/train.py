import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm

from model import SRCNN
def train(
        model,
        device,
        train_dir:Path,
        epochs:int=20,
        batch_size:int=16,
        learning_rate:float=1e-4,
        save_checkpoint:bool=True,
        scale_factor:int=3,
        dir_checkpoint="checkpoint/",
        debug=False
):
    return


def main(args):
    return

def get_args():
    parser = argparse.ArgumentParser(description="Train the SRCNN")
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--eval-dir', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale-factor', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_arguemnt('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
