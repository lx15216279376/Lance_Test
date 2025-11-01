import lancedb, numpy as np, time, tqdm, matplotlib.pyplot as plt
import random

def sample_query_vectors(tbl, n=1000):
    """用 search 逐条采样，避免一次性拉大块数据"""
    pool = []
    # 先取一条作为“种子”向量
    seed = tbl.search().limit(1).to_pandas().iloc[0]["emb"]
    pool.append(np.array(seed, dtype=np.float32))

    # 后续用随机 offset 跳页采样（避免重复）
    for idx in random.sample(range(1, tbl.count_rows()), k=n-1):
        row = tbl.search().limit(1).offset(idx).to_pandas().iloc[0]
        pool.append(np.array(row["emb"], dtype=np.float32))
    return pool

def bench(table_name: str):
    """
    对单张表进行多轮并发压测，返回 QPS 与 P99 延迟序列
    参数:
        table_name : str  待测表名（如 wiki_10k）
    返回:
        (concurrency, qps_list, p99_list)
          concurrency : list[int]  并发度序列
          qps_list    : list[float] 对应并发度下的 QPS
          p99_list    : list[float] 对应并发度下的 P99 延迟（毫秒）
    """
    db = lancedb.connect("/mnt/mydata/lx/lancedb")
    # concurrency = [1, 2, 4, 8, 16, 32]      # 模拟并发度
    concurrency = [1]      # 模拟并发度
    duration = 60                             # 每档的压力测试时长（秒）
    warmup = 3                                # 预热时长（秒）
    nprobes = 64                             # 与建索引时相同
    # 1. 打开表并抽取1000条真实向量作为查询向量
    tbl = db.open_table(table_name)
    query_pool = sample_query_vectors(tbl, n=1000)

    # 2. 初始化结果容器
    qps_list, p99_list = [], []

    # 3. 对每一档并发度进行压测
    for clients in concurrency:
        latencies = []          # 单次请求延迟列表（秒）
        # 预热
        _start = time.time()
        while time.time() - _start < warmup:
            _ = tbl.search(query_pool[0]).limit(10).nprobes(nprobes).to_pandas()
        
        # 正式压测
        start = time.time()     # 本轮压测开始时间
        sent = 0                # 已发请求计数

        # 3.1 模拟 clients 并发：单进程循环发请求
        #     每个循环 = 1 次 ANN 搜索
        cnt = 10
        while time.time() - start < duration:   # 压测固定时长
            query_vec = random.choice(query_pool)
            t0 = time.time()
            # 执行一次 Top-10 向量搜索
            _ = (tbl.search(query_vec)
                   .limit(10)          # 返回 10 条最相似结果
                   .nprobes(nprobes)   # 探测分区数，与建索引时一致
                   .to_pandas())       # 拉取结果（触发真正计算）
            latencies.append(time.time() - t0)  # 记录单次延迟（秒）
            sent += 1
        # 3.2 计算本轮指标
        elapsed = time.time() - start
        qps = sent / elapsed                  # 总请求数 / 总时长
        p99 = np.quantile(latencies, 0.99) * 1000  # P99 延迟（毫秒）
        print(f"Table: {table_name}, Clients: {clients}, Sent: {sent}")
        print(f"  QPS: {qps:.2f}, P99 Latency: {p99:.2f} ms")
        qps_list.append(qps)
        p99_list.append(p99)

    return concurrency, qps_list, p99_list

if __name__ == "__main__":
    tables = ["wiki_10k", "wiki_100k", "wiki_500k", "wiki_1M","wiki_3M"]
    # tables = ["wiki_10k"]
    # 1. 只跑一次 bench，缓存结果
    data = {tbl: bench(tbl) for tbl in tables}   # {表名: (clients, qps, p99)}

