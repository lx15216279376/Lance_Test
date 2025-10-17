# train_cnn_dali.py
import os, csv, time
import torch
import torch.nn as nn
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from tqdm import tqdm
from torch.multiprocessing import set_start_method

# ---------- 超参 ----------
BATCH   = 64                 # DALI 建议 256+
EPOCHS  = 5
LR      = 1e-3
DEVICE  = "cuda:0"
IMG_DIR = "/mnt/mydata/lx/docker_datasets/random_1000000_256/shared_1000000_256/train"
CSV_FILE= "./labels.csv"
CHKPT   = "../checkpoints/cnn/cnn_dali.pt"
os.makedirs(os.path.dirname(CHKPT), exist_ok=True)

# ---------- 读取标签 ----------
def load_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)                 # 跳过表头
        return [(row[0], int(row[1])) for row in reader]

# ---------- DALI Pipeline ----------
@pipeline_def(batch_size=BATCH, num_threads=1, device_id=0, seed=42)
def dali_pipeline():
    # 1. 从 CSV 读取文件名 & 标签
    jpg, label = fn.readers.file(
        file_root=IMG_DIR,
        files=[fname for fname, _ in load_csv(CSV_FILE)],
        labels=[lbl for _, lbl in load_csv(CSV_FILE)],
        name="Reader",
        random_shuffle=True,
    )
    # 1. 解码 → RGB → GPU
    images = fn.decoders.image(jpg, device="mixed", output_type=types.RGB)
    # 2. resize/crop
    images = fn.resize(images, resize_x=64, resize_y=64)
    # 3. 转置：NHWC → NCHW （关键！）
    images = fn.transpose(images, perm=[2, 0, 1])
    # 4. 归一化
    images = fn.cast(images, dtype=types.FLOAT) / 255.0
    return images, label

# ---------- CNN（同 PyTorch）----------
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
     # ① 先读 CSV 拿到总样本数
    csv_samples = load_csv(CSV_FILE)          # 返回 [(fname, label), ...]
    total_samples = len(csv_samples)

    model = CNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        # ② 每 epoch 新建一个 DALI 迭代器
        pipe = dali_pipeline()
        pipe.build()
        dali_iter = DALIGenericIterator(pipe, 
                                        output_map=["data", "label"],
                                        # size=total_samples,
                                        reader_name="Reader",
                                        auto_reset=True)
        model.train()
        tot_loss, tot_acc, n = 0., 0., 0
        total_batches = (total_samples + BATCH - 1) // BATCH   # 向上取整
        tic = time.time()
        for data in tqdm(dali_iter,total=total_batches,desc=f"Epoch {epoch+1}"):
            x = data[0]["data"]          # 已在 GPU
            y = data[0]["label"].squeeze().long().to(DEVICE)
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
        # dali_iter.reset()
        # del dali_iter, pipe
        
    torch.save(model.state_dict(), CHKPT)
    print("✅ DALI 训练完成，权重已保存")


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    train()