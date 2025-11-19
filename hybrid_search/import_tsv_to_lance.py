import os
import csv
import pandas as pd
import pyarrow as pa
import lancedb
from tqdm import tqdm

"""
将 TSV 文件导入 LanceDB 数据库。
支持流式读取大文件，避免内存溢出。
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "huggingkg_tiny")
TRIPLES_TSV = os.path.join(DATA_DIR, "triples.tsv")
TRIPLES_TSV_SAMPLE = os.path.join(DATA_DIR, "triples_sample_10k.tsv")

# LanceDB 数据库路径
LANCEDB_PATH = "/home/liuxuan/LanceTest/Lance/hybrid_search/lanceDB"
TABLE_NAME_FULL = "triples"
TABLE_NAME_SAMPLE = "triples_sample_10k"


def import_tsv_to_lancedb(tsv_path: str, db_path: str, table_name: str, batch_size: int = 100_000):
    """
    流式读取 TSV 文件，分批写入 LanceDB。
    
    Args:
        tsv_path: TSV 文件路径
        db_path: LanceDB 数据库路径
        table_name: 表名
        batch_size: 每批处理的行数
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"未找到文件: {tsv_path}")
    
    # 先统计总行数（用于进度条）
    total_rows = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        # 跳过表头
        next(f)
        total_rows = sum(1 for _ in f)
    
    print(f"📊 总行数: {total_rows:,}")
    
    # 连接到 LanceDB
    db = lancedb.connect(db_path)
    
    # 流式读取并分批写入
    batch_data = []
    first_batch = True
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        
        for row in tqdm(reader, total=total_rows, desc="读取并写入 TSV"):
            batch_data.append({
                "head_type": row["head_type"],
                "head": row["head"],
                "relation": row["relation"],
                "tail_type": row["tail_type"],
                "tail": row["tail"],
            })
            
            # 达到批次大小时，写入 LanceDB
            if len(batch_data) >= batch_size:
                df = pd.DataFrame(batch_data)
                
                if first_batch:
                    # 第一批：创建表（覆盖模式）
                    tbl = db.create_table(table_name, data=df, mode="overwrite")
                    first_batch = False
                else:
                    # 后续批次：追加数据
                    tbl.add(df)
                
                batch_data = []
        
        # 处理最后一批
        if batch_data:
            df = pd.DataFrame(batch_data)
            if first_batch:
                tbl = db.create_table(table_name, data=df, mode="overwrite")
            else:
                tbl.add(df)
    
    # 验证写入结果
    if not first_batch or batch_data:
        print(f"📝 写入完成...")
        print(f"✅ 已写入表: {table_name}")
        print(f"   总行数: {tbl.count_rows():,}")
        print(f"   列: {', '.join(tbl.schema.names)}")
    else:
        print("⚠️  没有数据可写入")


def verify_lancedb_table(db_path: str, table_name: str, num_rows: int = 5):
    """验证 LanceDB 表，显示前几行"""
    if not os.path.exists(db_path):
        print(f"❌ 数据库不存在: {db_path}")
        return
    
    db = lancedb.connect(db_path)
    
    if table_name not in db.table_names():
        print(f"❌ 表不存在: {table_name}")
        return
    
    tbl = db[table_name]
    print(f"\n📖 验证表: {table_name}")
    print(f"   总行数: {tbl.count_rows():,}")
    print(f"   列: {', '.join(tbl.schema.names)}")
    
    # 显示前几行
    df = tbl.head(num_rows).to_pandas()
    print(f"\n前 {num_rows} 行数据:")
    print(df.to_string(index=False))


def main():
    """主函数"""
    print("=" * 60)
    print("将 TSV 文件导入 LanceDB 数据库")
    print(f"数据库路径: {LANCEDB_PATH}")
    print("=" * 60)
    
    # 确保数据库目录存在
    os.makedirs(LANCEDB_PATH, exist_ok=True)
    
    # 导入样本文件
    # if os.path.exists(TRIPLES_TSV_SAMPLE):
    #     print(f"\n1️⃣  导入样本文件: {TRIPLES_TSV_SAMPLE}")
    #     import_tsv_to_lancedb(TRIPLES_TSV_SAMPLE, LANCEDB_PATH, TABLE_NAME_SAMPLE, batch_size=10_000)
    #     verify_lancedb_table(LANCEDB_PATH, TABLE_NAME_SAMPLE)
    
    # 导入全量文件
    # if os.path.exists(TRIPLES_TSV):
    #     file_size = os.path.getsize(TRIPLES_TSV)
    #     if file_size > 0:
    #         print(f"\n2️⃣  导入全量文件: {TRIPLES_TSV}")
    #         print(f"   文件大小: {file_size / 1024 / 1024:.2f} MB")
    #         import_tsv_to_lancedb(TRIPLES_TSV, LANCEDB_PATH, TABLE_NAME_FULL, batch_size=100_000)
    #         verify_lancedb_table(LANCEDB_PATH, TABLE_NAME_FULL, num_rows=5)
    #     else:
    #         print(f"\n⚠️  全量文件为空，跳过: {TRIPLES_TSV}")
    # else:
    #     print(f"\n⚠️  全量文件不存在，跳过: {TRIPLES_TSV}")
    
    # print("\n" + "=" * 60)
    # print("✅ 导入完成！")
    # print(f"数据库位置: {LANCEDB_PATH}")
    # print("=" * 60)
    verify_lancedb_table(LANCEDB_PATH, TABLE_NAME_FULL, num_rows=5)


if __name__ == "__main__":
    main()

