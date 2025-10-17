# jpg_to_lance.py
import os
import lance
import pyarrow as pa
from tqdm import tqdm
import re
from lance import dataset as ld
IMG_DIR = "/mnt/mydata/lx/docker_datasets/random_500000_256/shared_500000_256/train"        # ① 改成你的文件夹
LANCE_OUT = "/mnt/mydata/lx/lance_dataset/random_500000_256.lance"             # ② 输出文件

def make_batches():
    for fname in tqdm(os.listdir(IMG_DIR)):
        if not fname.lower().endswith(('.jpg', '.jpeg')):
            continue
        with open(os.path.join(IMG_DIR, fname), 'rb') as f:
            binary = f.read()
        yield pa.RecordBatch.from_arrays(
            [
                pa.array([fname], pa.string()),
                pa.array([binary], pa.binary()),
            ],
            ["image_name", "image"],
        )

def download_lance():
    schema = pa.schema([
        pa.field("image_name", pa.string()),
        pa.field("image", pa.binary()),
    ])

    reader = pa.RecordBatchReader.from_batches(schema, make_batches())
    ld.write_dataset(reader, LANCE_OUT, schema)
    print(f"✅ 已写入 {LANCE_OUT}")

# 给随机数据集添加标签列
def add_label():
    LANCE_PATH = LANCE_OUT  # 原文件
    # 1. 读取原表，逐行添加标签
    # 1. 读取原表，生成新列数据（int64）
    ds = ld.LanceDataset(LANCE_PATH)
    labels = []
    for row in tqdm(ds.to_table().to_pylist()):
        num = int(re.findall(r'\d+', row["image_name"])[0])
        labels.append(num % 10)

    # 2. 零拷贝加列（只写新列，旧列不动）
    new_column = pa.table({"label": labels})
    ds.add_columns(new_column)      # 关键 API

    print("✅ 零拷贝加列完成，最新版本号：", ds.latest_version)

# 添加label列和row_id列并创建索引
def create_index():
    ds = ld.LanceDataset(LANCE_OUT)
    # 1. 若尚无 row_id 列，先追加（int64）

    if "row_id" not in ds.schema.names:
        batches = []
        for idx in tqdm(range(ds.count_rows())):
            batches.append(pa.RecordBatch.from_arrays([pa.array([idx], pa.int64())], ["row_id"]))
        ds.add_columns(pa.RecordBatchReader.from_batches(pa.schema([("row_id", pa.int64())]), batches))

    # 2. 创建标量索引（实验 API）
    ds.create_scalar_index(
        column="row_id",
        index_type="BTREE",        # 关键：指定标量索引
        name="rowid_BTREE"
    )

    print("✅ 标量 BTREE 索引已创建")

if __name__ == "__main__":
    download_lance()
    add_label()
    # create_index()
