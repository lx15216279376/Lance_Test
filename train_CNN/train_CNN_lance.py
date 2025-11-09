# train_cnn_lance.py
import lance, cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.multiprocessing import set_start_method
import time, collections
import os

# ----- 超参 -----
BATCH   = 64
EPOCHS  = 5
LR      = 1e-3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LANCE_FILE = "/mnt/mydata/lx/lance_dataset/random_100000_256.lance"   

_timer = collections.defaultdict(float)  # 全局累加器
_worker_stats = {}  # 每个 worker 独享

# -----  Dataset -----
class LanceImageDS(Dataset):
    def __init__(self, lance_path,transform=None):
        self.ds = lance.dataset(lance_path)
        print(type(self.ds))
        self.transform = transform

    def __len__(self):
        return self.ds.count_rows()

    def __getitem__(self, idx):
        # 添加了统计时间开销
        wid = os.getpid()                      # 用进程号当 key
        if wid not in _worker_stats:
            # _worker_stats[wid] = {'cnt': 0, 'take': 0., 'pylist': 0., 'decode': 0.,'cv_ops': 0., 'torch': 0.}
            _worker_stats[wid] = {'cnt': 0, 'take': 0., 'process': 0.}

        # 1. take 开销
        t0 = time.perf_counter()
        tbl = self.ds.take([idx])

        row = tbl.to_pylist()[0]

        img_b = row["image"]
        label = row["label"]
        _worker_stats[wid]['take'] += time.perf_counter() - t0
       

        # 2. process开销
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.IMREAD_COLOR)    # 把 JPG 二进制解码成 BGR 数组（HWC，uint8）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))          # 统一尺寸
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        _worker_stats[wid]['process'] += time.perf_counter() - t0

        _worker_stats[wid]['cnt'] += 1
        cnt = _worker_stats[wid]['cnt']
        if cnt % 10000 == 0:                      # 每 1000 样本打印一次
            # print(_worker_stats)
            total = sum([_worker_stats[wid][k] for k in ['take', 'process']])
            print(f"[worker {wid}][sample {cnt}] "
                  f"take:{_worker_stats[wid]['take']/total*100:.1f}% "
                  f"process:{_worker_stats[wid]['process']/total*100:.1f}% "
                  f"avg={total/cnt*1000:.2f} ms")
            
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
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

    for epoch in range(EPOCHS):
        model.train()
        tot_loss, tot_acc, n = 0., 0., 0
        tic = time.perf_counter() # 计时开始
        for step,(x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tot_loss += loss.item() * x.size(0)
            tot_acc  += (logits.argmax(1) == y).sum().item()
            n += x.size(0)

        toc = time.perf_counter() # 计时结束
        throughput = n / (toc - tic) # 样本吞吐率
        print(f"Epoch {epoch+1}: loss={tot_loss/n:.4f}  acc={tot_acc/n:.3f} throughput={throughput:.1f} samples/sec")
        # print(f"Epoch {epoch+1}: throughput={throughput:.1f} samples/sec")
    
    torch.save(model.state_dict(), "../checkpoints/cnn/cnn_random.pt")
    print("✅ 训练完成，权重已保存")

if __name__ == "__main__":
    set_start_method("spawn", force=True)   # ① 必须在主模块最先执行
    train()