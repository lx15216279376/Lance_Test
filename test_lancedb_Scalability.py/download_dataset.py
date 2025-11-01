import lancedb, pyarrow as pa, pandas as pd, glob, math
import pyarrow.dataset as ds
from tqdm import tqdm

db = lancedb.connect("/mnt/mydata/lx/lancedb")   # 本地数据库目录

schema = pa.schema([
    ("id", pa.int64()),
    ("title", pa.string()),
    ("text", pa.string()),
    ("emb", pa.list_(pa.float32(), 1024))
])

def unify(vec, target_len: int = 1024):
    """截断或 zero-pad 到 target_len，返回 list[float32]"""
    if vec is None:
        vec = []
    vec = list(vec)                      # 可能传入 numpy array
    if len(vec) < target_len:
        vec.extend([0.0] * (target_len - len(vec)))
    return vec[:target_len]

def write_table(name: str,
                total_rows: int,
                file_pattern: str = "/mnt/mydata/lx/tmp/wikipedia-22-12-en-embeddings/data/train-*.parquet",
                batch_rows: int = 20_000):
    """流式读 Parquet → 统一长度 → 批量写 LanceDB"""
    tbl = db.create_table(name, schema=schema, mode="overwrite")
    written = 0
    pbar = tqdm(total=total_rows, desc=name)

    for file in glob.glob(file_pattern):
        if written >= total_rows:
            break
        # PyArrow 逐块读
        dataset = ds.dataset(file, format="parquet")
        for batch in dataset.to_batches(batch_size=batch_rows, columns=["title", "text", "emb"]):
            need = min(batch.num_rows, total_rows - written)
            if need <= 0:
                break
            batch = batch.slice(0, need)
            df = batch.to_pandas()

            # 统一长度（快速 apply）
            df["emb"] = df["emb"].apply(unify)
            # 重新编号
            df["id"] = range(written, written + len(df))

            tbl.add(df)
            written += len(df)
            pbar.update(len(df))
    pbar.close()
    print(f"{name} 实际写入 {tbl.count_rows()} 条")

sizes = {
    # "wiki_10k":   10_000,
    # "wiki_100k": 100_000,
    # "wiki_500k": 500_000,
    # "wiki_1M":  1_000_000,
    "wiki_3M":  3_000_000,
}

for tbl_name, rows in sizes.items():
    write_table(tbl_name, rows)

for t in db.table_names():
    print(t, db.open_table(t).count_rows())

