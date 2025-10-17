# generate_labels.py
import os, re, csv
from tqdm import tqdm

IMG_DIR   = "/mnt/mydata/lx/docker_datasets/random_500000_256/shared_500000_256/train"   # 你的原始图片文件夹
CSV_OUT   = os.path.join(".", "labels.csv")  # 输出路径

# 支持的后缀
EXTS = (".jpeg", ".jpg", ".png")

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "label"])          # 表头

    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(EXTS)]
    for fname in tqdm(files, desc="生成标签"):
        # 提取文件名里的数字 → mod 10 当伪标签
        digits = re.findall(r'\d+', fname)
        label  = int(digits[0]) % 10 if digits else 0
        writer.writerow([fname, label])

print(f"✅ 已生成 {CSV_OUT} ，共 {len(files)} 条记录")