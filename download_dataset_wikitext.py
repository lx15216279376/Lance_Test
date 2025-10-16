import lance
import pyarrow as pa

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm  # optional for progress tracking

def tokenize(sample, field='text'):
    return tokenizer(sample[field])['input_ids']

# def tokenize(sample, field='text'):
#     return tokenizer(sample[field], truncation=True, max_length=1024)['input_ids']

def process_samples(dataset, num_samples=100_000, field='text'):
    current_sample = 0
    for sample in tqdm(dataset, total=num_samples):
        # If we have added all 5M samples, stop
        if current_sample == num_samples:
            break
        if not sample[field]:
            continue
        # Tokenize the current sample
        tokenized_sample = tokenize(sample, field)
        # Increment the counter
        current_sample += 1
        # Yield a PyArrow RecordBatch
        yield pa.RecordBatch.from_arrays(
            [tokenized_sample], 
            names=["input_ids"]
        )

if __name__ == "__main__":
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', streaming=True)['train']
    # dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')['train']
    dataset = dataset.shuffle(seed=1337)

    # 定义lance数据集的模式（schema）
    schema = pa.schema([
        pa.field("input_ids", pa.int64())   # 只保存token序列
    ])

    # 创建一个PyArrow RecordBatchReader，按需生成批次
    reader = pa.RecordBatchReader.from_batches(
        schema, 
        process_samples(dataset, num_samples=500_000, field='text') # For 500K samples
    )

    # Write the dataset to disk
    lance.write_dataset(
        reader, 
        "wikitext_500K.lance",
        schema
    )