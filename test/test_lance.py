import shutil
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import duckdb
from lance.vector import vec_to_table
import struct
import time

def download_sift_tolance():
    uri = "vec_data.lance"

    with open("sift/sift_base.fvecs", mode="rb") as fobj:
        buf = fobj.read()
        data = np.array(struct.unpack("<128000000f", buf[4 : 4 + 4 * 1000000 * 128])).reshape((1000000, 128))
        dd = dict(zip(range(1000000), data))

    table = vec_to_table(dd)
    lance.write_dataset(table, uri, max_rows_per_group=8192, max_rows_per_file=1024*1024)

def search_without_index():
    uri = "vec_data.lance"
    sift1m = lance.dataset(uri)

    samples = duckdb.query("SELECT vector FROM sift1m USING SAMPLE 100").to_df().vector
    # print(samples)

    tot = 0
    for q in samples:
        start = time.time()
        tbl = sift1m.to_table(nearest={"column": "vector", "q": q, "k": 10})
        end = time.time()
        tot += (end - start)

    print(f"Avg(sec): {tot / len(samples)}")
    print(tbl.to_pandas())

def create_index():
    uri = "vec_data.lance"
    sift1m = lance.dataset(uri)

    # 创建IVF_PQ索引
    sift1m.create_index(
        "vector",
        index_type="IVF_PQ", # specify the IVF_PQ index type
        num_partitions=256,  # IVF
        num_sub_vectors=16,  # PQ
    )

def search_with_index():
    uri = "vec_data.lance"
    sift1m = lance.dataset(uri)

    samples = duckdb.query("SELECT vector FROM sift1m USING SAMPLE 100").to_df().vector
    # print(samples)

    sift1m = lance.dataset(uri)
    tot = 0
    for q in samples:
        start = time.time()
        tbl = sift1m.to_table(nearest={"column": "vector", "q": q, "k": 10})
        end = time.time()
        tot += (end - start)

    print(f"Avg(sec): {tot / len(samples)}")
    print(tbl.to_pandas())

if __name__ == "__main__":
    download_sift_tolance()
    search_without_index()
    create_index()
    search_with_index()


