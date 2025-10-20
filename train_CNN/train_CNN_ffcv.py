import os, time, torch, torch.nn as nn
from tqdm import tqdm
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, RandomHorizontalFlip, ToTorchImage, NormalizeImage
from ffcv.fields.decoders import IntDecoder
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
from torch.cuda.amp import autocast, GradScaler
import numpy as np

BATCH   = 64
EPOCHS  = 5
LR      = 1e-3
DEVICE = torch.device("cuda:0")
BETON   = "/mnt/mydata/lx/ffcv_dataset/random_1000000_256.beton"
CHKPT   = "../checkpoints/cnn/cnn_ffcv.pt"
os.makedirs(os.path.dirname(CHKPT), exist_ok=True)

IMAGENET_MEAN = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
IMAGENET_STD  = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

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
        x = x / 255.0
        return self.classifier(self.features(x))

def build_loader(beton_path, batch_size, shuffle=True):
    image_pipeline = [
        RandomResizedCropRGBImageDecoder((64, 64)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(DEVICE, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),  # 必须指定 dims，避免 numba 报错
        ToDevice(DEVICE, non_blocking=True)
    ]
    loader = Loader(
        beton_path,
        batch_size=batch_size,
        num_workers=1,
        order=OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL,
        pipelines={
            "image": image_pipeline,
            "label": label_pipeline
        },
        drop_last=False,
        batches_ahead=2,
        os_cache=True,
    )
    return loader

def train():
    loader = build_loader(BETON, BATCH, shuffle=True)
    model = CNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        tot_loss, tot_acc, n = 0., 0., 0
        tic = time.time()
        for x, y in tqdm(loader, total=len(loader), desc=f"Epoch {epoch+1}"):
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
    print("✅ FFCV 训练完成，权重已保存")

if __name__ == "__main__":
    train()