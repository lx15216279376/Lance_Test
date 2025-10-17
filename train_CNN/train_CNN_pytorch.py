# train_cnn_pytorch.py
import os, csv, time, cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.multiprocessing import set_start_method

# ---------- 超参 ----------
BATCH   = 64
EPOCHS  = 5
LR      = 1e-3
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "/mnt/mydata/lx/docker_datasets/random_500000_256/shared_500000_256/train"          # 原始 JPG 文件夹
CSV_FILE= "./labels.csv"   # 标签文件
CHKPT   = "../checkpoints/cnn/cnn_pytorch.pt"

os.makedirs(os.path.dirname(CHKPT), exist_ok=True)

# ---------- 原始数据集 ----------
class JpgCsvDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir   = img_dir
        self.transform = transform
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            self.samples = [(row[0], int(row[1])) for row in csv.reader(f)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- 数据增强 ----------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ---------- CNN（同 Lance 版本）----------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---------- 训练 & 计时 ----------
def train():
    ds = JpgCsvDataset(IMG_DIR, CSV_FILE, transform=train_transform)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=1,
                        persistent_workers=True, pin_memory=True)
    model = CNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        tot_loss, tot_acc, n = 0., 0., 0
        tic = time.time()
        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            tot_loss += loss.item() * x.size(0)
            tot_acc  += (logits.argmax(1) == y).sum().item()
            n += x.size(0)

        toc = time.time()
        print(f"Epoch {epoch+1}: loss={tot_loss/n:.4f}  acc={tot_acc/n:.3f}  "
              f"throughput={n/(toc-tic):.1f} samples/sec")

    torch.save(model.state_dict(), CHKPT)
    print("✅ PyTorch 标准训练完成，权重已保存")

if __name__ == "__main__":
    # set_start_method("spawn", force=True)
    train()