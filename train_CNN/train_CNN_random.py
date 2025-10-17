# train_cnn_lance.py
import lance, cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.multiprocessing import set_start_method
import time
from lance import dataset as ld

# ----- 超参 -----
BATCH   = 64
EPOCHS  = 5
LR      = 1e-3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LANCE_FILE = "/mnt/mydata/lx/lance_dataset/random_1000000_256.lance"   # ① 改成你的文件

# -----  Dataset -----
class LanceImageDS(Dataset):
    def __init__(self, lance_path, transform=None):
        self.ds = ld.LanceDataset(lance_path)
        self.transform = transform

    def __len__(self):
        return self.ds.count_rows()

    def __getitem__(self, idx):
        row = self.ds.take([idx]).to_pylist()[0]
        img_b = row["image"]
        label = row["label"]
        # 二进制 → CV 图像
        img = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))          # 统一尺寸
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if self.transform:
            img = self.transform(img)
        return img, label

# ----- CNN (LeNet-like) -----
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

# ----- 训练循环 -----
def train():
    ds = LanceImageDS(LANCE_FILE)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=1,pin_memory=True, persistent_workers=True)
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        tot_loss, tot_acc, n = 0., 0., 0
        tic = time.perf_counter() # 计时开始
        sample_cnt = 0
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
            sample_cnt += x.size(0)

        toc = time.perf_counter() # 计时结束
        throughput = sample_cnt / (toc - tic) # 样本吞吐率
        print(f"Epoch {epoch+1}: loss={tot_loss/n:.4f}  acc={tot_acc/n:.3f} throughput={throughput:.1f} samples/sec")
        # print(f"Epoch {epoch+1}: throughput={throughput:.1f} samples/sec")
    
    torch.save(model.state_dict(), "../checkpoints/cnn/cnn_random.pt")
    print("✅ 训练完成，权重已保存")

if __name__ == "__main__":
    set_start_method("spawn", force=True)   # ① 必须在主模块最先执行
    train()