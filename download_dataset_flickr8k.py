import os
import cv2
import random

import lance
import pyarrow as pa

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

captions = "Flickr8k/Flickr8k.token.txt"    # 标题文件
image_folder = "Flickr8k/Flicker8k_Dataset/"    # 图片文件夹

with open(captions, "r") as fl:
    annotations = fl.readlines() 

# Converts the annotations where each element of this list is a tuple consisting of image file name, caption number and caption itself
# 将注释转换为一个列表，其中每个元素都是一个包含图像文件名、标题编号和标题本身的元组
annotations = list(map(lambda x: tuple([*x.split('\t')[0].split('#'), x.split('\t')[1]]), annotations))

# 将注释按图像ID分组，每个图像对应多个标题，并按标题编号排序
captions = []
image_ids = set(ann[0] for ann in annotations)  # 取出所有图片文件名，去重
for img_id in tqdm(image_ids):                  # 遍历每个图片文件名，内层循环找到当前图片对应的所有标题
    current_img_captions = []
    for ann_img_id, num, caption in annotations:
        if img_id == ann_img_id:
            current_img_captions.append((num, caption))

    # Sort by the annotation number
    current_img_captions.sort(key=lambda x: x[0])   # 按标题编号排序
    captions.append((img_id, tuple([x[1] for x in current_img_captions])))  # 只保留标题文本，丢弃标题编号

# 上面处理得到的记录如下：
# [('1000268201_693b08cb0e.jpg',  ('cap0', 'cap1', 'cap2', 'cap3', 'cap4')),...]


# Process()是一个PyArrow RecordBatch生成器
def process(captions):
    for img_id, img_captions in tqdm(captions):
        try:
            with open(os.path.join(image_folder, img_id), 'rb') as im:
                binary_im = im.read()   # 以二进制格式读取图片

        except FileNotFoundError:
            print(f"img_id '{img_id}' not found in the folder, skipping.")
            continue
        
        # 构造3个PyArrow数组，分别存放图片ID、图片二进制数据和描述列表
        img_id = pa.array([img_id], type=pa.string())
        img = pa.array([binary_im], type=pa.binary())
        capt = pa.array([img_captions], pa.list_(pa.string(), -1))
        # 把3列拼成一行，返回一个RecordBatch
        yield pa.RecordBatch.from_arrays(
            [img_id, img, capt], 
            ["image_id", "image", "captions"]
        )

# 定义lance数据集的模式（schema）
schema = pa.schema([
    pa.field("image_id", pa.string()),
    pa.field("image", pa.binary()),
    pa.field("captions", pa.list_(pa.string(), -1)),
])


reader = pa.RecordBatchReader.from_batches(schema, process(captions))
lance.write_dataset(reader, "flickr8k.lance", schema)