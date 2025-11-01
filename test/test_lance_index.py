import lancedb
from datetime import timedelta
from datasets import load_dataset
import pyarrow as pa
import numpy as np

# Connect to LanceDB
uri = "../data/sample-lancedb"
db = lancedb.connect(uri)

def ingest_data():
    sample_dataset = load_dataset(
        "sunhaozhepy/ag_news_sbert_keywords_embeddings", split="test[:5000]"
    )
    vector_dim = len(sample_dataset[0]["keywords_embeddings"])
    print(sample_dataset.column_names)
    # print(sample_dataset[:5])

    table_name = "lancedb-cloud-quickstart"
    table = db.create_table(table_name, data=sample_dataset, mode="overwrite")

    # convert list to fixedsizelist on the vector column
    table.alter_columns(
        dict(path="keywords_embeddings", data_type=pa.list_(pa.float32(), vector_dim))
    )

def create_index_IVF():
    table_name = "lancedb-cloud-quickstart"
    table = db.open_table(table_name)
    print(table.count_rows())
    table.create_index(metric="cosine", vector_column_name="keywords_embeddings")
    print(table.search(np.random.random((768))).limit(2).nprobes(20).refine_factor(10).to_pandas())

def create_index_HNSW():
    table_name = "lancedb-cloud-quickstart"
    table = db.open_table(table_name)
    print(table.count_rows())
    table.create_index(
        vector_column_name="keywords_embeddings",
        index_type="IVF_HNSW_SQ",          # 字符串即可
        metric="l2",            # 或 "l2"
        m=16,                       # 邻接表大小，默认 16
        ef_construction=200,        # 建索引时动态候选集，默认 200
    )
    print(table.search(np.random.random((768))).limit(2).to_pandas())

def create_index_Binary():
    table_name = "test-hamming"
    ndim = 256
    schema = pa.schema([
        pa.field("id", pa.int64()),
        # For dim=256, store every 8 bits in a byte (32 bytes total)
        pa.field("vector", pa.list_(pa.uint8(), 32)),
    ])
    table = db.create_table(table_name, schema=schema, mode="overwrite")

    data = []
    for i in range(1024):
        vector = np.random.randint(0, 2, size=ndim)
        vector = np.packbits(vector)  # Optional: pack bits to save space
        data.append({"id": i, "vector": vector})
    table.add(data)

    table.create_index(
        metric="hamming",
        vector_column_name="vector",
        index_type="IVF_FLAT"
    )

    query = np.random.randint(0, 2, size=256)
    query = np.packbits(query)
    df = table.search(query).metric("hamming").limit(10).to_pandas()
    df.vector = df.vector.apply(np.unpackbits)

if __name__ == "__main__":
    # ingest_data()
    # create_index_IVF()
    # create_index_HNSW()
    # create_index_Binary()
    import time

    table_name = "lancedb-cloud-quickstart"
    table = db.open_table(table_name)
    index_name = "keywords_embeddings_idx"
    table.wait_for_index([index_name])
    print(table.index_stats(index_name))