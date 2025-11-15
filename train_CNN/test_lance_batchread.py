# bench_take_batch.py
import lance, numpy as np, time, os
from tqdm import tqdm

DATA_PATH = "/mnt/mydata/lx/lance_dataset/random_1000000_256_plus.lance"  # ① 改这里
SIZES = [1, 64, 128, 256, 512, 1024]
ITERS = 1000

def bench(size):
    ds = lance.dataset(DATA_PATH)
    total_take_ms = 0.0
    total_samples = 0
    # 预生成足够多随机行号（非连续）
    # indices = np.random.randint(0, ds.count_rows(), size * ITERS, dtype=np.int32)

    pbar = tqdm(range(ITERS), desc=f"take({size})")
    for i in pbar:
        idx = np.random.choice(ds.count_rows(), size, replace=False).astype(np.int32)
        t0 = time.perf_counter()
        tbl = ds.take(idx)
        dur_ms = (time.perf_counter() - t0) * 1000
        total_take_ms += dur_ms
        total_samples += size
        if i % 100 == 0 and i > 0:
            pbar.set_postfix(avg=f"{total_take_ms / total_samples:.2f} ms/sample")

    avg_sample = total_take_ms / total_samples
    avg_take = total_take_ms / ITERS
    print(f"take({size:3d})  ——  avg per sample: {avg_sample:.3f} ms  |  avg take call: {avg_take:.3f} ms")


if __name__ == "__main__":
    for s in SIZES:
        bench(s)