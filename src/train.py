import argparse
import logging
import os
import copy
from tqdm import tqdm

# torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# custom
from model import SRCNN
from dataset import TrainDataset, EvalDataset
from utils import AverageMeter, psnr

def train(args):
    # set up device, instantiate the SRCNN model, set up criterion and optimizer
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.patch_mapping.parameters()},
        {'params': model.non_linear_mapping.parameters()},
        {'params': model.reconstruction.parameters(), 'lr':args.lr * .1}
    ], lr=args.lr)

    # (Initialize logging)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'''Starting training:
        Epoch:          {args.num_epochs}
        Batch size:     {args.batch_size}
        Learning rate:  {args.lr}
        Scale:          {args.scale}
        Device:         {device.type}
    ''')

    # configure datasets and dataloaders
    train_dataset = TrainDataset(args.train_file)
    eval_dataset = EvalDataset(args.eval_file)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # track best parameters and values
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset)- len(train_dataset)% args.batch_size)) as pbar:
            pbar.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for batch in train_dataloader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = criterion(preds,labels)
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        epoch_psnr = AverageMeter()
        for batch in eval_dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad(): preds = model(inputs).clamp(0.,1.)
            epoch_psnr.update(psnr(preds,labels), len(inputs))
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))    
    return

def get_args():
    # setting up argumentparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str)
    parser.add_argument('--eval-file', type=str)
    parser.add_argument('--outputs-dir', type=str)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    return args

if __name__ == "__main__":
    args = get_args()
    train(args)
    