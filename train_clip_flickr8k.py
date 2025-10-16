import cv2
import lance
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from transformers import AutoModel, AutoTokenizer

import itertools
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')

class Config:
    img_size = (128, 128)
    bs = 32
    head_lr = 1e-3
    img_enc_lr = 1e-4
    text_enc_lr = 1e-5
    max_len = 18
    img_embed_dim = 2048
    text_embed_dim = 768
    projection_dim = 256
    temperature = 1.0
    num_epochs = 5
    img_encoder_model = 'resnet50'
    text_encoder_model = 'bert-base-cased'

def load_image(ds, idx):
    # Utility function to load an image at an index and convert it from bytes format to img format
    # 按索引从Lance读出二进制图片，解码成图片格式
    raw_img = ds.take([idx], columns=['image']).to_pydict()
    raw_img = np.frombuffer(b''.join(raw_img['image']), dtype=np.uint8)
    img = cv2.imdecode(raw_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def load_caption(ds, idx):
    # Utility function to load an image's caption. Currently we return the longest caption of all
    # 取图片描述，简单起见只取最长的描述
    captions = ds.take([idx], columns=['captions']).to_pydict()['captions'][0]
    return max(captions, key=len)

# 自定义Lance数据集类，用于加载图片及其对应的描述
class CLIPLanceDataset(Dataset):
    """Custom Dataset to load images and their corresponding captions"""
    def __init__(self, lance_path, max_len=18, tokenizer=None, transforms=None):
        self.ds = lance.dataset(lance_path)
        self.max_len = max_len
        # Init a new tokenizer if not specified already
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') if not tokenizer else tokenizer
        self.transforms = transforms

    def __len__(self):
        return self.ds.count_rows()

    def __getitem__(self, idx):
        # Load the image and caption
        img = load_image(self.ds, idx)
        caption = load_caption(self.ds, idx)

        # Apply transformations to the images
        # 对图片做变换
        if self.transforms:
            img = self.transforms(img)

        # Tokenize the caption
        # 对描述做tokenize
        caption = self.tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        # Flatten each component of tokenized caption otherwise they will cause size mismatch errors during training
        caption = {k: v.flatten() for k, v in caption.items()}

        return img, caption

class ImageEncoder(nn.Module):
    """Encodes the Image"""
    def __init__(self, model_name, pretrained = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, img):
        return self.backbone(img)

class TextEncoder(nn.Module):
    """Encodes the Caption"""
    def __init__(self, model_name):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, captions):
        output = self.backbone(**captions)
        return output.last_hidden_state[:, 0, :]

class Head(nn.Module):
    """Projects both into Embedding space"""
    # 两层MLP做投影，将图片和文本的特征投影到同一维度
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected

        return self.layer_norm(x)

# 计算对比损失 
def loss_fn(img_embed, text_embed, temperature=0.2):
    """
    https://arxiv.org/abs/2103.00020/
    """
    # Calculate logits, image similarity and text similarity
    logits = (text_embed @ img_embed.T) / temperature
    img_sim = img_embed @ img_embed.T
    text_sim = text_embed @ text_embed.T
    # Calculate targets by taking the softmax of the similarities
    targets = F.softmax(
        (img_sim + text_sim) / 2 * temperature, dim=-1
    )
    img_loss = (-targets.T * nn.LogSoftmax(dim=-1)(logits.T)).sum(1)
    text_loss = (-targets * nn.LogSoftmax(dim=-1)(logits)).sum(1)
    return (img_loss + text_loss) / 2.0

def forward(img, caption):
    # Transfer to device
    # 图片和描述张量都放到GPU上
    img = img.to('cuda')
    for k, v in caption.items():
        caption[k] = v.to('cuda')

    # Get embeddings for both img and caption
    # 将图片和描述分别通过各自的编码器和投影头，得到它们的嵌入表示
    img_embed = img_head(img_encoder(img))
    text_embed = text_head(text_encoder(caption))

    return img_embed, text_embed


if __name__ == "__main__":
    # 定义数据增强操作  
    train_augments = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(Config.img_size),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Define image encoder, image head, text encoder, text head and a tokenizer for tokenizing the caption
    img_encoder = ImageEncoder(model_name=Config.img_encoder_model).to('cuda')
    img_head = Head(Config.img_embed_dim, Config.projection_dim).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(Config.text_encoder_model)
    text_encoder = TextEncoder(model_name=Config.text_encoder_model).to('cuda')
    text_head = Head(Config.text_embed_dim, Config.projection_dim).to('cuda')

    # Since we are optimizing two different models together, we will define parameters manually
    parameters = [
        {"params": img_encoder.parameters(), "lr": Config.img_enc_lr},
        {"params": text_encoder.parameters(), "lr": Config.text_enc_lr},
        {
            "params": itertools.chain(
                img_head.parameters(),
                text_head.parameters(),
            ),
            "lr": Config.head_lr,
        },
    ]

    optimizer = torch.optim.Adam(parameters)

    # We assume the flickr8k.lance dataset is in the same directory
    dataset = CLIPLanceDataset(
        lance_path="flickr8k.lance",
        max_len=Config.max_len,
        tokenizer=tokenizer,
        transforms=train_augments
    )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=Config.bs,
        pin_memory=True
    )

    # 将所有模型切换到训练模式
    img_encoder.train()
    img_head.train()
    text_encoder.train()
    text_head.train()

    # 模型保存路径
    save_dir = "checkpoints/clip_flickr8k/"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(Config.num_epochs):
        print(f"{'='*20} Epoch: {epoch+1} / {Config.num_epochs} {'='*20}")

        prog_bar = tqdm(dataloader)
        for img, caption in prog_bar:
            optimizer.zero_grad(set_to_none=True)

            img_embed, text_embed = forward(img, caption)
            loss = loss_fn(img_embed, text_embed, temperature=Config.temperature).mean()

            loss.backward()
            optimizer.step()


            prog_bar.set_description(f"loss: {loss.item():.4f}")
        print()

        # 保存模型，只存关键状态，省空间
        state = {
            "img_encoder":  img_encoder.state_dict(),
            "img_head":     img_head.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "text_head":    text_head.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "epoch":        epoch,
            "loss":         loss.item(),
        }
        save_path = os.path.join(save_dir, f"clip_flickr8k_ep{epoch+1}.pt")
        torch.save(state, save_path)
        print(f"✅ 已保存 epoch{epoch+1} 权重")