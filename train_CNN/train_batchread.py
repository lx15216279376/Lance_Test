# train_cnn_lance_batch.py
# 改造点：take([idx]) → take(batch_indices)  一次 64 张
import lance, cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.multiprocessing import set_start_method
import time, collections, os, torch

# ---------- 超参 ----------
BATCH = 64                       # 一次下发 64 张
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LANCE_FILE = "/mnt/mydata/lx/lance_dataset/random_1000000_256_plus.lance"   # ① 改这里

_worker_stats = {}               # 每个 worker 独享计时

# ---------- Dataset：批量 take + 统一解码 ----------
class LanceBatchDS(Dataset):
    def __init__(self, lance_path, transform=None):
        self.ds = lance.dataset(lance_path)
        self.order = np.random.permutation(self.ds.count_rows())
        self.transform = transform

    def __len__(self):
        return self.ds.count_rows() // BATCH  # 整除，避免尾部

    def __getitem__(self, idx):
        # 0. 构造 batch 索引（手动映射：第 i 个请求 → [i*B, (i+1)*B)）
        base = idx * BATCH
        # indices = np.arange(base, base + BATCH) % self.ds.count_rows()
        indices = self.order[base:base + BATCH]   # 连续切片，但顺序已全局 shuffle
        indices = indices.astype(np.int32)

        # 1. 批量 take
        wid = os.getpid()
        if wid not in _worker_stats:
            _worker_stats[wid] = {'cnt': 0, 'take': 0., 'process': 0.}
        t0 = time.perf_counter()
        tbl = self.ds.take(indices.tolist())          # ← 一次 take 64 行
        _worker_stats[wid]['take'] += time.perf_counter() - t0

        # 2. 统一解码
        t0 = time.perf_counter()
        rows = tbl.to_pylist()
        imgs_b = [r["image"] for r in rows]
        labels = [r["label"] for r in rows]

        tensors = []
        for img_b in imgs_b:
            img = cv2.imdecode(np.frombuffer(img_b, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            tensors.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0)
        batch_tensor = torch.stack(tensors)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        _worker_stats[wid]['process'] += time.perf_counter() - t0

        # 3. 统计
        # _worker_stats[wid]['cnt'] += BATCH
        # cnt = _worker_stats[wid]['cnt']
        # if cnt % 10000 == 0:
        #     total = sum([_worker_stats[wid][k] for k in ['take', 'process']])
        #     print(f"[worker {wid}][sample {cnt}] "
        #           f"take:{_worker_stats[wid]['take']/total*100:.1f}% "
        #           f"process:{_worker_stats[wid]['process']/total*100:.1f}% "
        #           f"avg={total/cnt*1000:.2f} ms")

        if self.transform:
            batch_tensor = self.transform(batch_tensor)
        # 返回一个 **batch**，所以 DataLoader 的 batch_size 必须设为 None
        return batch_tensor, label_tensor

# ---------- CNN ----------
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

# ---------- 训练 ----------
def train():
    ds = LanceBatchDS(LANCE_FILE)
    loader = DataLoader(ds,
                        batch_size=None,           # Dataset 已打包
                        shuffle=False,             # 我们在 Dataset 里手动映射
                        num_workers=1,
                        pin_memory=True,
                        persistent_workers=True)

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

    for epoch in range(EPOCHS):
        model.train()
        tot_loss = tot_acc = n = 0
        tic = time.perf_counter()
        for step, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
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

        toc = time.perf_counter()
        throughput = n / (toc - tic)
        print(f"Epoch {epoch+1}: loss={tot_loss/n:.4f}  "
              f"acc={tot_acc/n:.3f}  throughput={throughput:.1f} samples/sec")

    torch.save(model.state_dict(), "../checkpoints/cnn/cnn_random_batch.pt")
    print("✅ 训练完成，权重已保存 → ../checkpoints/cnn/cnn_random_batch.pt")

if __name__ == "__main__":
    set_start_method("spawn", force=True)
    train()