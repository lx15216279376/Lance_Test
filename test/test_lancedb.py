import lancedb
import pandas as pd
import pyarrow as pa
import polars as pl
import numpy as np
from lancedb.pydantic import Vector, LanceModel
from pydantic import BaseModel

uri = "data/sample-lancedb"
db = lancedb.connect(uri)
print(db)

def test_create_table_tuples():
  # 从元组或字典创建表
  data = [
      {"vector": [1.1, 1.2], "lat": 45.5, "long": -122.7},
      {"vector": [0.2, 1.8], "lat": 40.1, "long": -74.1},
  ]
  db.create_table("test_table", data)
  print(db["test_table"].head())

def test_create_table_DataFrame():
  # 从 Pandas DataFrame 创建表
  data = pd.DataFrame(
      {
          "vector": [[1.1, 1.2, 1.3, 1.4], [0.2, 1.8, 0.4, 3.6]],
          "lat": [45.5, 40.1],
          "long": [-122.7, -74.1],
      }
  )
  db.create_table("my_table_pandas", data)
  print(db["my_table_pandas"].head())

def test_create_table_schema():
  # 从 custom_schema 创建表
  custom_schema = pa.schema(
      [
          pa.field("vector", pa.list_(pa.float32(), 4)),
          pa.field("lat", pa.float32()),
          pa.field("long", pa.float32()),
      ]
  )
  data = pd.DataFrame({
        "vector": [[1.0, 2.0, 3.0, 4.0],
                   [5.0, 6.0, 7.0, 8.0]],
        "lat": [10.1, 20.2],
        "long": [30.3, 40.4],
    })
  tbl = db.create_table("my_table_custom_schema", data, schema=custom_schema)
  print(tbl)

def test_create_table_polars():
  # 从 Polars DataFrame 创建表
  data = pl.DataFrame(
      {
          "vector": [[1.1, 1.2, 1.3], [0.2, 1.8, 0.4]],
          "lat": [45.5, 40.1],
          "long": [-122.7, -74.1],
      }
  )
  db.create_table("my_table_polars", data)
  print(db["my_table_polars"].head())

def test_create_table_arrow():
  # 从 Arrow Table 创建表，包含 float16 向量
  dim = 16
  total = 2
  schema = pa.schema(
      [pa.field("vector", pa.list_(pa.float16(), dim)), pa.field("text", pa.string())]
  )
  data = pa.Table.from_arrays(
      [
          pa.array(
              [np.random.randn(dim).astype(np.float16) for _ in range(total)],
              pa.list_(pa.float16(), dim),
          ),
          pa.array(["foo", "bar"]),
      ],
      ["vector", "text"],
  )
  tbl = db.create_table("f16_tbl", data, schema=schema)
  print(tbl)

def test_create_table_pydantic():
  # 将pydantic模型作为模式创建表
  class Content(LanceModel):
    movie_id: int
    vector: Vector(128)
    genres: str
    title: str
    imdb_id: int

    @property
    def imdb_url(self) -> str:
        return f"https://www.imdb.com/title/tt{self.imdb_id}"


  tbl = db.create_table("movielens_small", schema=Content)
  print(tbl)

def test_create_table_nested():
  # 嵌套 pydantic 模型作为模式创建表
  class Document(BaseModel):
    content: str
    source: str

  class NestedSchema(LanceModel):
    id: str
    vector: Vector(1536)
    document: Document

  tbl = db.create_table("nested_table", schema=NestedSchema)
  print(tbl)
# 使用迭代器遍历大型数据集，一次性创建表
def test_create_table_LargeDatasets():
  def make_batches():
      for i in range(5):
          yield pa.RecordBatch.from_arrays(
              [
                  pa.array(
                      [[3.1, 4.1, 5.1, 6.1], [5.9, 26.5, 4.7, 32.8]],
                      pa.list_(pa.float32(), 4),
                  ),
                  pa.array(["foo", "bar"]),
                  pa.array([10.0, 20.0]),
              ],
              ["vector", "item", "price"],
          )

  schema = pa.schema(
      [
          pa.field("vector", pa.list_(pa.float32(), 4)),
          pa.field("item", pa.utf8()),
          pa.field("price", pa.float32()),
      ]
  )
  db.create_table("batched_tale", make_batches(), schema=schema)
  print(db['batched_tale'].head())

if __name__ == "__main__":
    # test_create_table_tuples()
    # test_create_table_DataFrame()
    # test_create_table_schema()
    # test_create_table_polars()
    # test_create_table_arrow()
    # test_create_table_pydantic()
    # test_create_table_nested()
    # test_create_table_LargeDatasets()
    print(db.table_names())
    tbl = db.open_table("test_table")