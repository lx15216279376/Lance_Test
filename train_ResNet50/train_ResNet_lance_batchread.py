#!/usr/bin/env python
# 训练入口：python train_cnn_lance_batch.py --data /path/to/lance_folder --batch-size 256 --epochs 5 --workers 12
import os, sys, time, argparse, shutil, math
import torch, torch.nn as nn
import torchvision
from torchvision.models import resnet18, resnet50
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader          # ← 普通 Dataset
import lance
from torchvision.transforms import (
    RandomResizedCrop, RandomHorizontalFlip, ColorJitter,
    CenterCrop, Resize, ToTensor, Normalize, Compose
)
from PIL import Image
import io
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ----------- ImageNet 均值方差 -----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
BATCH = 256

# ----------- 数据管道：Dataset + 批量 take ----------
class LanceImageNet(Dataset):
    def __init__(self, lance_path, split='train', crop=224):
        super().__init__()
        self.ds      = lance.dataset(lance_path)
        self.order = np.random.permutation(self.ds.count_rows())
        self.split   = split
        self.crop    = crop
        self._num_rows = self.ds.count_rows()

        # 与官方对齐的变换
        if split == 'train':
            self.transform = Compose([
                RandomResizedCrop(crop, scale=(0.08, 1.0)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        else:
            self.transform = Compose([
                Resize(256),
                CenterCrop(crop),
                ToTensor(),
                Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])

    # 每个 epoch 重新 shuffle 一次全表
    def resample(self):
        self.order = np.random.permutation(self.ds.count_rows())

    # ---------- 核心：随机 + 批量 take ----------
    def __getitem__(self, idx):
        # 0. 每段 64 行，全局已 shuffle（DataLoader 负责段顺序）
        base = idx * BATCH
        indices = self.order[base:base + BATCH]   # 连续切片，但顺序已全局 shuffle
        indices = indices.astype(np.int32)          # Lance 要求 int32

        # 1. 一次性 take 64 行
        tbl   = self.ds.take(indices.tolist())      # ← 批量 take
        rows  = tbl.to_pylist()
        imgs_b = [r["image"] for r in rows]
        labels = [r["label"] for r in rows]

        # 2. 统一解码 + 变换
        tensors = []
        for img_b in imgs_b:
            img = Image.open(io.BytesIO(img_b)).convert('RGB')
            img = self.transform(img)               # CHW Tensor
            tensors.append(img)
        images_tensor = torch.stack(tensors, dim=0)  # [64,3,224,224]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return images_tensor, labels_tensor

    def __len__(self):
        return self._num_rows // BATCH          # 整除，避免尾部


# ----------- DataLoader 工厂（多进程 + prefetch） -----------
def build_lance_loader(lance_path, batch_size, split='train', shuffle=False, num_workers=1):
    ds = LanceImageNet(lance_path, split=split)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True,
                      persistent_workers=True)


# ----------- 模型 / 指标 / 工具 -----------
def get_model(model_name='resnet50', num_classes=1000):
    if model_name == 'resnet18':
        model = resnet18(weights=None)
    elif model_name == 'resnet50':
        model = resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        nn.init.normal_(model.fc.weight, 0, 0.01)
        nn.init.constant_(model.fc.bias, 0)
    return model


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / target.size(0)).item() for k in topk]


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(loader, model, criterion, optimizer, epoch, device, scaler):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1     = AverageMeter('Acc@1', ':6.2f')
    top5     = AverageMeter('Acc@5', ':6.2f')

    pbar = tqdm(loader, total=len(loader), desc=f'Epoch {epoch:2d}', ncols=120)
    for images, target in pbar:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
        pbar.set_postfix(loss=f'{losses.avg:.3f}', top1=f'{top1.avg:.2f}%', top5=f'{top5.avg:.2f}%')
    return losses.avg, top1.avg


@torch.no_grad()
def validate(loader, model, criterion, device):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1     = AverageMeter('Acc@1', ':6.2f')
    top5     = AverageMeter('Acc@5', ':6.2f')

    pbar = tqdm(loader, total=len(loader), desc='Val  ', ncols=120, leave=False)
    for images, target in pbar:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.4f}')
    return top1.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='lance folder')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--workers', default=12, type=int, help='Number of data loading workers')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save-dir', default='./ckpts', type=str)
    parser.add_argument('--model', default='resnet50', choices=['resnet18', 'resnet50'], help='Model architecture')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting training with {args.model}...")
    best_acc1 = 0

    # 数据：lance 文件
    train_loader = build_lance_loader(os.path.join(args.data, 'train.lance'), args.batch_size, split='train', shuffle=False, num_workers=args.workers)
    val_loader   = build_lance_loader(os.path.join(args.data, 'val.lance'),   args.batch_size, split='val',   shuffle=False, num_workers=args.workers)

    # 模型
    model = get_model(args.model, num_classes=1000).to(device)
    if torch.cuda.device_count() > 1:
        print(f"=> Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt['epoch']
        best_acc1   = ckpt['best_acc1']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        print(f"=> loaded checkpoint '{args.resume}' (epoch {ckpt['epoch']})")

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

    for epoch in range(start_epoch, args.epochs):
        train_loader.dataset.resample()   # ② 每 epoch 重新洗牌
        train_loss, train_acc1 = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, scaler)
        val_acc1 = validate(val_loader, model, criterion, device)
        optimizer.step()
        scheduler.step()

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(args.save_dir, f'checkpoint_ep{epoch+1}.pth'))
        if is_best:
            shutil.copy(os.path.join(args.save_dir, f'checkpoint_ep{epoch+1}.pth'),
                        os.path.join(args.save_dir, 'best.pth'))
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Acc1', train_acc1, epoch)
        writer.add_scalar('Val/Acc1', val_acc1, epoch)

    writer.close()
    print(f" * Best Acc@1: {best_acc1:.3f}")


if __name__ == '__main__':
    main()