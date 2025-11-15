#!/usr/bin/env python
import os
import re
import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
from tqdm import tqdm
import lance


def iter_image_records(img_dir: Path, label_mod: int):
    """
    按文件逐个生成 Lance 记录，默认根据文件名中的数字对 label_mod 取模得到标签。
    你也可以根据自己的需求调整标签逻辑。
    """
    pattern = re.compile(r'\d+')
    for fname in tqdm(sorted(img_dir.iterdir()), desc=f"scan {img_dir.name}"):
        if not fname.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue
        with fname.open("rb") as f:
            binary = f.read()
        match = pattern.search(fname.stem)
        if match:
            label = int(match.group()) % label_mod
        else:
            label = 0
        yield {"image_name": fname.name, "image": binary, "label": label}


def write_lance_random(img_dir: Path, lance_uri: Path, batch_size: int = 10_000, label_mod: int = 10):
    """
    将随机图像数据写入 Lance 数据集。
    - img_dir: 存放随机图像的目录
    - lance_uri: 输出的 .lance 路径
    - batch_size: 每批写入条数，建议 10k 或更大以减少碎片
    - label_mod: 标签映射，默认对文件名数字取模
    """
    schema = pa.schema([
        ("image_name", pa.string()),
        ("image", pa.binary()),
        ("label", pa.int32())
    ])

    buf = []
    written = False
    for idx, record in enumerate(iter_image_records(img_dir, label_mod)):
        buf.append(record)
        if (idx + 1) % batch_size == 0:
            table = pa.Table.from_pylist(buf, schema=schema)
            lance.write_dataset(table, str(lance_uri), mode="append" if written else None)
            written = True
            buf.clear()

    if buf:
        table = pa.Table.from_pylist(buf, schema=schema)
        lance.write_dataset(table, str(lance_uri), mode="append" if written else None)

    print(f"✅ 已写入 Lance 数据集：{lance_uri}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="随机图像所在目录")
    parser.add_argument("--output", required=True, help="输出 .lance 文件路径")
    parser.add_argument("--batch-size", type=int, default=10_000, help="写入批大小")
    parser.add_argument("--label-mod", type=int, default=10, help="标签取模基数")
    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    lance_uri = Path(args.output)
    lance_uri.parent.mkdir(parents=True, exist_ok=True)

    write_lance_random(img_dir, lance_uri, args.batch_size, args.label_mod)


if __name__ == "__main__":
    main()