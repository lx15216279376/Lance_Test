import os
import lancedb
import pandas as pd
import math
import lance
import pyarrow as pa
import av
import cv2,numpy as np

VIDEO_DIR = "train_val"          # 你的文件夹
LANCEDB_PATH = "/mnt/mydata/lx/lancedb" # 数据库目录
TABLE_NAME = "raw_videos"

# 存入lancedb
def save_videos_to_lancedb():
    """读取文件夹下所有 mp4 视频，存入 LanceDB"""
    # 扫描所有 mp4
    videos = [v for v in os.listdir(VIDEO_DIR) if v.lower().endswith(".mp4")]
    assert videos, f"在 {VIDEO_DIR} 未找到 mp4"

    # 读成二进制
    records = []
    for idx, fname in enumerate(videos):
        with open(os.path.join(VIDEO_DIR, fname), "rb") as f:
            records.append({"id": idx, "video_data": f.read()})

    # 写入 LanceDB
    db = lancedb.connect(LANCEDB_PATH)
    tbl = db.create_table(TABLE_NAME, data=pd.DataFrame(records))
    print(f"✅ 已存入 {len(tbl)} 条原始视频，总大小 {tbl.to_pandas()['video_data'].apply(len).sum() / 1024 / 1024:.1f} MB")

# 把目录下所有 mp4 以 Blob 形式写入 Lance（large_binary）
def test_save_video_blob():
    VIDEO_DIR = "./train_val"          # 包含若干 *.mp4 的文件夹
    LANCE_PATH = "./tmp/video_blob.lance"   # 输出 Lance 数据集
    # 1. 收集文件
    mp4_files = [os.path.join(VIDEO_DIR, f)
                 for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    if not mp4_files:
        raise FileNotFoundError("no mp4 found in " + VIDEO_DIR)

    # 2. 读成字节
    labels = list(range(len(mp4_files)))          # 简单用序号当标签
    blobs = [open(p, "rb").read() for p in mp4_files]

    # 2. 关键：用 metadata 把列标记为 blob
    schema = pa.schema([
        pa.field("label", pa.int64()),
        pa.field("video_data", pa.large_binary(),
                metadata={"lance-encoding:blob": "true"})  # ← 核心
    ])

    tbl = pa.Table.from_pydict(
        {"label": labels, "video_data": blobs},
        schema=schema
    )

    # 4. 写入 Lance
    lance.write_dataset(tbl, LANCE_PATH, mode="overwrite")
    print(f"✅ {len(blobs)} 个视频已写入 {LANCE_PATH}")

def test_read_video_blob():
    # 打开 Lance 数据集（路径含 .lance）
    ds = lance.dataset("./tmp/video_blob.lance")
    # 起始 / 结束时间（毫秒）
    start_time, end_time = 500, 1000

    # 取出第 5 行的 "video" blob 列，返回 List[BlobFile]（类文件对象）
    blobs = ds.take_blobs(blob_column="video_data",ids=[5])

    # 把 blob 当作视频文件直接喂给 PyAV，无需落盘
    with av.open(blobs[0]) as container:
        # 选择第一条视频流
        stream = container.streams.video[0]
        # 解码时跳过非关键帧，加速定位
        stream.codec_context.skip_frame = "NONKEY"

        # 将毫秒时间戳转换为流内部时间基（ticks）
        start_time = start_time / stream.time_base
        # 取整数部分，符合 FFmpeg 时间戳格式
        start_time = start_time.as_integer_ratio()[0]
        end_time = end_time / stream.time_base

        # 跳转到最近关键帧（起始点附近）
        container.seek(start_time, stream=stream)

        fps = float(stream.average_rate)

        # 从 seek 点开始逐帧解码
        for frame in container.decode(stream):
            # 超过结束时间立即终止
            if frame.time > end_time:
                break
            # 展示视频帧
            img = frame.to_ndarray(format="bgr24")
            cv2.imshow("clip", img)
             # 每帧等待时间 = 帧时长（毫秒）
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    # save_videos_to_lancedb()
    # test_save_video_blob()
    test_read_video_blob()