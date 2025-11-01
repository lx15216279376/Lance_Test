import lancedb, time, numpy as np

db = lancedb.connect("/mnt/mydata/lx/lancedb")
tbl = db.open_table("wiki_3M")

# df = tbl.search().limit(10_000).to_pandas()
# unique = df["emb"].apply(tuple).nunique()
# print("唯一向量数:", unique, "/ 10000")

# 建立 IVF_PQ 索引
tbl.create_index(
    metric="cosine",
    index_type="IVF_PQ",
    vector_column_name="emb",
    num_partitions=64,      
    num_sub_vectors=64,     # 1024 维可被 64 整除
)


# 建立 HNSW 索引
# tbl.create_index(
#     vector_column_name="emb",
#     index_type="IVF_HNSW_SQ",
#     m=16,                # 邻接表大小
#     ef_construction=200  # 建索引时动态候选集
# )
# 预计 3–6 秒完成

# 随机取一条做查询向量
row = tbl.search().limit(1).to_pandas().iloc[0]
query = np.array(row["emb"], dtype=np.float32)

t0 = time.time()
df = (tbl
      .search(query)
      .limit(10)
      .nprobes(20)         # IVF 时有效
      .to_pandas())
print("耗时:", (time.time() - t0) * 1000, "ms")
print(df[["id", "_distance"]])