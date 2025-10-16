import lancedb
import numpy as np
import pyarrow as pa
import os

# Connect to LanceDB Cloud/Enterprise
uri = "db://lancedb-demo-lx-fa0l56"
api_key = "sk_VUSD3SANQZEC5JMLUIPKLE2TOB2I5KGEI4CEFZHSKIQPEA5QJ3WQ===="
region = "us-east-1"

# (Optional) For LanceDB Enterprise, set the host override to your enterprise endpoint
host_override = os.environ.get("LANCEDB_HOST_OVERRIDE")

db = lancedb.connect(
  uri=uri,
  api_key=api_key,
  region=region,
  host_override=host_override
)

# from datasets import load_dataset

# # Load a sample dataset from HuggingFace with pre-computed embeddings
# sample_dataset = load_dataset("sunhaozhepy/ag_news_sbert_keywords_embeddings", split="test[:1000]")
# print(f"Loaded {len(sample_dataset)} samples")
# print(f"Sample features: {sample_dataset.features}")
# print(f"Column names: {sample_dataset.column_names}")

# # Preview the first sample
# print(sample_dataset[0])

# # Get embedding dimension
# vector_dim = len(sample_dataset[0]["keywords_embeddings"])
# print(f"Embedding dimension: {vector_dim}")

# import pyarrow as pa

# # Create a table with the dataset
# table_name = "lancedb-cloud-test"
# table = db.create_table(table_name, data=sample_dataset, mode="overwrite")

# # Convert list to fixedsizelist on the vector column
# table.alter_columns(dict(path="keywords_embeddings", data_type=pa.list_(pa.float32(), vector_dim)))
# print(f"Table '{table_name}' created successfully")

from datetime import timedelta

table = db.open_table("lancedb-cloud-test")
# Create a vector index and wait for it to complete
table.create_index("cosine", vector_column_name="keywords_embeddings", wait_timeout=timedelta(seconds=1200))
print(table.index_stats("keywords_embeddings_idx"))



