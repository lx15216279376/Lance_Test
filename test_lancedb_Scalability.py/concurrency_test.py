#!/usr/bin/env python3
import lancedb
import numpy as np
import time
import random
import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
import threading

# ========== 参数 ==========
DB_PATH      = "/mnt/mydata/lx/lancedb"
TABLES       = ["wiki_10k", "wiki_100k", "wiki_500k", "wiki_1M"]
CONCURRENCY  = [1, 2, 4, 8, 16, 32]   # 并发度列表
DURATION     = 10                     # 每档压测秒数
WARMUP       = 2                      # 预热秒数
TOP_K        = 10
NPROBES      = 64                     # 与建索引时一致
SAMPLE_POOL  = 1000                   # 预采样查询向量数
# ==========================

def sample_query_vectors(tbl, n=1000):
    """用 search().limit + offset 随机采样，避免 Arrow 大块读"""
    pool = []
    # 先采 1 条种子
    pool.append(np.array(tbl.search().limit(1).to_pandas().iloc[0]["emb"], dtype=np.float32))
    # 其余随机跳页
    total_rows = tbl.count_rows()
    for idx in random.sample(range(1, total_rows), k=n-1):
        row = tbl.search().limit(1).offset(idx).to_pandas().iloc[0]
        pool.append(np.array(row["emb"], dtype=np.float32))
    return pool

def real_concurrent_bench(table_name, clients, duration=10, warmup=2):
    """
    真并发压测（clients=1 时退化为顺序循环）
    返回: (qps, p99_ms)
    """
    db  = lancedb.connect(DB_PATH)
    tbl = db.open_table(table_name)
    query_pool = sample_query_vectors(tbl, SAMPLE_POOL)

    latencies = []
    counter   = 0
    lock      = threading.Lock()

    def single_query():
        nonlocal counter
        qv = random.choice(query_pool)
        t0 = time.time()
        _ = tbl.search(qv).limit(TOP_K).nprobes(NPROBES).to_pandas()
        lat = time.time() - t0
        with lock:
            latencies.append(lat)
            counter += 1

    # ---------- 预热 ----------
    end = time.time() + warmup
    while time.time() < end:
        single_query()

    # ---------- 正式压测 ----------
    latencies.clear()
    counter = 0
    start   = time.time()

    if clients == 1:
        # 顺序循环（单线程）
        with tqdm.tqdm(total=None, desc=f"{table_name} C={clients}") as pbar:
            while time.time() - start < duration:
                single_query()
                pbar.update(1)
    else:
        # 多线程真并发
        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
            futures = []
            while time.time() - start < duration:
                futures.append(pool.submit(single_query))
            concurrent.futures.wait(futures)

    qps   = counter / (time.time() - start)
    p99_ms = np.quantile(latencies, 0.99) * 1000
    return qps, p99_ms

def main():
    # 1. 跑所有表 + 所有并发度
    results = {}          # {表: {并发: (qps, p99)}}
    for tbl in TABLES:
        results[tbl] = {}
        for c in tqdm.tqdm(CONCURRENCY, desc=tbl):
            qps, p99 = real_concurrent_bench(tbl, c, DURATION, WARMUP)
            results[tbl][c] = (qps, p99)
            print(f"{tbl} C={c}: QPS={qps:.2f}  P99={p99:.2f} ms")

    # 2. 画图
    plt.figure(figsize=(12, 4))
    # 2.1 QPS
    plt.subplot(1, 2, 1)
    for tbl in TABLES:
        x = CONCURRENCY
        y = [results[tbl][c][0] for c in x]
        plt.plot(x, y, marker='o', label=tbl)
    plt.xlabel("Concurrent Clients")
    plt.ylabel("QPS")
    plt.title("QPS vs. Concurrency")
    plt.grid()
    plt.legend()

    # 2.2 P99
    plt.subplot(1, 2, 2)
    for tbl in TABLES:
        x = CONCURRENCY
        y = [results[tbl][c][1] for c in x]
        plt.plot(x, y, marker='o', label=tbl)
    plt.xlabel("Concurrent Clients")
    plt.ylabel("P99 Latency (ms)")
    plt.title("P99 Latency vs. Concurrency")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("scalability_all.png", dpi=300)
    print("Saved -> scalability_all.png")

if __name__ == "__main__":
    main()