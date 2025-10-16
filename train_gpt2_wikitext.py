import numpy as np
import lance

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# We'll be training the pre-trained GPT2 model in this example
# 使用预训练的gpt2模型作为基础模型
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Also define some hyperparameters
# 定义一些超参数
lr = 3e-4
nb_epochs = 1
block_size = 1024
batch_size = 8
device = 'cuda:0'
dataset_path = 'wikitext_500K.lance'

# 根据索引从Lance数据集中获取对应数据
def from_indices(dataset, indices):
    """Load the elements on given indices from the dataset"""
    chunk = dataset.take(indices).to_pylist()
    chunk = list(map(lambda x: x['input_ids'], chunk))
    return chunk

# 基于Lance数据集的自定义数据集类
class LanceDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        block_size,
    ):
        # Load the lance dataset from the saved path
        # 加载lance格式的数据集
        self.ds = lance.dataset(dataset_path)
        self.block_size = block_size

        # Doing this so the sampler never asks for an index at the end of text
        # 避免采样器采样到文件末尾
        self.length = self.ds.count_rows() - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Generate a window of indices starting from the current idx to idx+block_size
        and return the tokens at those indices
        """
        # 从idx开始，取block_size个连续的样本
        window = np.arange(idx, idx + self.block_size)
        sample = from_indices(self.ds, window)

        return {"input_ids": torch.tensor(sample), "labels": torch.tensor(sample)}

# 自定义采样器 ，需要确保每个样本之间至少间隔block_size个token
class LanceSampler(Sampler):
    r"""Samples tokens randomly but `block_size` indices apart.

    Args:
        data_source (Dataset): dataset to sample from
        block_size (int): minimum index distance between each random sample
    """

    def __init__(self, data_source, block_size=512):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.available_indices = list(range(0, self.num_samples, block_size))
        np.random.shuffle(self.available_indices)

    def __iter__(self):
        yield from self.available_indices

    def __len__(self) -> int:
        return len(self.available_indices)
    
if __name__ == "__main__":
    # Define the dataset, sampler and dataloader
    dataset = LanceDataset(dataset_path, block_size)
    sampler = LanceSampler(dataset, block_size)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True
    )

    # Define the optimizer, training loop and train the model!
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(nb_epochs):
        print(f"========= Epoch: {epoch+1} / {nb_epochs} =========")
        epoch_loss = []
        prog_bar = tqdm(dataloader, total=len(dataloader))
        for batch in prog_bar:
            optimizer.zero_grad(set_to_none=True)

            # Put both input_ids and labels to the device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Perform one forward pass and get the loss
            outputs = model(**batch)
            loss = outputs.loss

            # Perform backward pass
            loss.backward()
            optimizer.step()

            prog_bar.set_description(f"loss: {loss.item():.4f}")

            epoch_loss.append(loss.item())

    # Calculate training perplexity for this epoch
    try:
        perplexity = np.exp(np.mean(epoch_loss))
    except OverflowError:
        perplexity = float("-inf")

    print(f"train_perplexity: {perplexity}")