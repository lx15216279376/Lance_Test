# build_ffcv.py
import os, csv, sys
import ffcv
from ffcv.fields import RGBImageField, IntField
from ffcv.writer import DatasetWriter
import cv2, numpy as np

# ---------- 参数 ----------
IMG_DIR   = "/mnt/mydata/lx/docker_datasets/random_1000000_256/shared_1000000_256/train"
CSV_FILE  = "./labels_1000000.csv"
BETON_OUT = "/mnt/mydata/lx/ffcv_dataset/random_1000000_256.beton"
BATCH_WRITE = 100   # 每块写入样本数

# ---------- 读取 CSV ----------
def load_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        return [(row[0], int(row[1])) for row in reader]

# ---------- 可索引数据集 ----------
class CsvImageList:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        with open(os.path.join(IMG_DIR, fname), "rb") as f:
            buf = f.read()
        
        # 0.3rc1 要求：np.ndarray HWC uint8
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, label   # np.ndarray uint8

# ---------- 主函数 ----------
def main():
    samples = load_csv(CSV_FILE)
    print(f"共 {len(samples)} 张图，开始生成 FFCV 二进制 ...")
    writer = DatasetWriter(
        BETON_OUT,
        {
            'image': RGBImageField(write_mode='jpg',   # 保持原 JPG 压缩
                                max_resolution=None,
                                compress_probability=1.0,
                                jpeg_quality=90),
            'label': IntField(),
        },
        num_workers=16,  # 并行 IO
    )
    writer.from_indexed_dataset(
        CsvImageList(samples),
        chunksize=BATCH_WRITE,
    )
    print(f"✅ 已生成 {BETON_OUT}")

if __name__ == "__main__":
    main()

