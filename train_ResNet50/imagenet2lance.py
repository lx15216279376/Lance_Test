#!/usr/bin/env python
import os
import json
import lance
import pyarrow as pa
from pathlib import Path
from tqdm import tqdm
from lance import dataset as ld
import pandas as pd

IMAGENET_DIR = Path('/mnt/mydata/lx/ILSVRC2012')   # 改成你的
OUT_DIR      = Path('/mnt/mydata/lx/lance_dataset/ILSVRC2012_lance')
BATCH_WRITE  = 10_000   # 每 1w 张写一次磁盘
OUT_DIR.mkdir(exist_ok=True)

def build_label_map(train_dir):
    return {d.name: idx for idx, d in enumerate(sorted(d for d in train_dir.iterdir() if d.is_dir()))}

def write_lance_split(split: str, label_map: dict):
    src_dir = IMAGENET_DIR / split
    lance_uri = OUT_DIR / f'{split}.lance'
    if lance_uri.exists():
        print(f'👉 {lance_uri} 已存在，跳过'); return

    # 1. 扫描文件列表（同之前）
    if split == 'train':
        records = [(str(p), label_map[p.parent.name])
                   for synset_dir in tqdm(list(src_dir.iterdir()), desc='scan')
                   if synset_dir.is_dir()
                   for p in synset_dir.glob('*.JPEG')]
    else:  # val
        df = pd.read_csv(IMAGENET_DIR / 'val_id.csv', names=['path', 'synset'],skiprows=1)
        records = [(str(IMAGENET_DIR / row[0]), label_map[row[1]])
               for row in df.itertuples(index=False)]

    schema = pa.schema([('image', pa.binary()), ('label', pa.int32())])
    buf = []
    print(f'🚀 流式转换 {split} -> {lance_uri}')
    for idx, (img_path, lbl) in enumerate(tqdm(records, desc=f'convert {split}')):
        with open(img_path, 'rb') as f:
            buf.append({'image': f.read(), 'label': lbl})
        if (idx + 1) % BATCH_WRITE == 0:
            table = pa.Table.from_pylist(buf, schema=schema)

            if not lance_uri.exists():
                ld.write_dataset(table, str(lance_uri))
            else:
                ld.write_dataset(table, str(lance_uri), mode='append')
            buf.clear()

    if buf:
        table = pa.Table.from_pylist(buf, schema=schema)
        ld.write_dataset(table, str(lance_uri), mode='append')
    print(f'✅ {split} 完成，共 {len(records)} 张')

if __name__ == '__main__':
    label_map = build_label_map(IMAGENET_DIR / 'train')
    (OUT_DIR / 'label_map.json').write_text(json.dumps(label_map))
    write_lance_split('train', label_map)
    write_lance_split('val', label_map)
    print('🎉 全部转换完成！')