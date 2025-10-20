#!/usr/bin/env python
import os, time, argparse, shutil
import torch, torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm   # <-- 新增
from torch.amp import autocast, GradScaler

def get_model(num_classes=1000):
    model = resnet50(weights=None)
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / target.size(0)))
        return res

def train_one_epoch(loader, model, criterion, optimizer, epoch, device, args, writer,scaler):
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
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 实时刷新进度条后缀
        pbar.set_postfix(loss=f'{losses.avg:.3f}',
                         top1=f'{top1.avg:.2f}%',
                         top5=f'{top5.avg:.2f}%')

    if writer:
        writer.add_scalar('Train/Loss', losses.avg, epoch)
        writer.add_scalar('Train/Acc1', top1.avg, epoch)

def validate(loader, model, criterion, device, args, writer, epoch):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1     = AverageMeter('Acc@1', ':6.2f')
    top5     = AverageMeter('Acc@5', ':6.2f')

    pbar = tqdm(loader, total=len(loader), desc='Val  ', ncols=120, leave=False)
    with torch.no_grad():
        for images, target in pbar:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast('cuda'):
                output = model(images)
                loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            pbar.set_postfix(loss=f'{losses.avg:.3f}',
                             top1=f'{top1.avg:.2f}%',
                             top5=f'{top5.avg:.2f}%')

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.4f}')
    if writer:
        writer.add_scalar('Val/Loss', losses.avg, epoch)
        writer.add_scalar('Val/Acc1', top1.avg, epoch)
    return top1.avg

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data', help='ImageNet root')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--workers', default=12, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save-dir', default='./ckpts', type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_acc1 = 0

    # 数据
    train_tf, val_tf = get_transforms()
    train_set = datasets.ImageFolder(os.path.join(args.data, 'train'), train_tf)
    val_set   = datasets.ImageFolder(os.path.join(args.data, 'val'),   val_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # 模型
    model = get_model()
    if torch.cuda.device_count() > 1:
        print(f"=> Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
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
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args, writer,scaler)
        acc1 = validate(val_loader, model, criterion, device, args, writer, epoch)
        scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
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
    writer.close()

if __name__ == '__main__':
    main()