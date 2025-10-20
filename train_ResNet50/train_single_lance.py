#!/usr/bin/env python
import os, sys
import os, time, argparse, shutil
import torch, torch.nn as nn
import torchvision
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import IterableDataset, DataLoader
import lance
from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, CenterCrop, Resize, ToTensor, Normalize
from PIL import Image
import io
from lance import dataset as ld

# ----------- Lance 数据管道 -----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class LanceImageNet(IterableDataset):
    def __init__(self, lance_path, batch_size, split='train', crop=224):
        super().__init__()
        self.ds        = lance.dataset(lance_path)
        self.batch_size= batch_size
        self.split     = split
        self.crop      = crop
        # 用类接口，内部自动处理随机状态
        if split == 'train':
            self.transform = Compose([
                Resize(256),                      # 先缩到 256
                RandomResizedCrop(crop, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
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

    def __iter__(self):
        for batch in self.ds.to_batches(columns=['image', 'label'], batch_size=self.batch_size):
            images_bytes = batch.column('image').to_pylist()
            labels       = torch.from_numpy(batch.column('label').to_numpy()).long()
            for i in range(len(images_bytes)):
                img = Image.open(io.BytesIO(images_bytes[i])).convert('RGB')
                img = self.transform(img)          # 用类接口，兼容性最好
                yield img, labels[i]

def build_lance_loader(lance_path, batch_size, split='train', shuffle=False, drop_last=False, num_workers=0):
    ds = LanceImageNet(lance_path, batch_size, split, shuffle, drop_last)
    # Lance 已内部 batch，这里 batch_size=None
    return DataLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=True)

# ----------- 训练逻辑 -----------
def get_model(num_classes=1000):
    model = resnet50(weights=None)
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / target.size(0)).item() for k in topk]

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(loader, model, criterion, optimizer, epoch, device, scaler):
    model.train()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    pbar = tqdm(loader,  desc=f'Epoch {epoch}', ncols=120)
    for images, target in pbar:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
        pbar.set_postfix(loss=f'{losses.avg:.3f}', top1=f'{top1.avg:.2f}%')
    return losses.avg, top1.avg

@torch.no_grad()
def validate(loader, model, criterion, device):
    model.eval()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    for images, target in tqdm(loader, total=len(loader), desc='Val  ', ncols=120, leave=False):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
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
    print("Starting training...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='lance folder')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--save-dir', default='./ckpts', type=str)
    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = build_lance_loader(os.path.join(args.data, 'train.lance'),
                                      args.batch_size, split='train', shuffle=True)
    val_loader   = build_lance_loader(os.path.join(args.data, 'val.lance'),
                                      args.batch_size, split='val', shuffle=False)

    model = get_model().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

    best_acc1 = 0
    for epoch in range(args.epochs):
        train_loss, train_acc1 = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, scaler)
        val_acc1 = validate(val_loader, model, criterion, device)
        scheduler.step()

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                   os.path.join(args.save_dir, f'ckpt_ep{epoch+1}.pt'))
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Acc1', train_acc1, epoch)
        writer.add_scalar('Val/Acc1', val_acc1, epoch)
        writer.close()

      
if __name__ == '__main__': 
    main()