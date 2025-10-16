import os
import shutil
import lance
import pyarrow as pa
import torch
from collections import OrderedDict

# 定义保存模型参数的全局模式（schema）
GLOBAL_SCHEMA = pa.schema(
    [
        pa.field("name", pa.string()),
        # pa.field("value", pa.list_(pa.float64(), -1)),
        pa.field("value", pa.list_(pa.float32(), -1)), # 存为float32，节省空间
        pa.field("shape", pa.list_(pa.int64(), -1)), # Is a list with variable shape because weights can have any number of dims
    ]
)

# 获取模型的状态字典后，遍历每个权重，将其展平，然后返回权重名称、扁平权重和权重的原始形状
# def _save_model_writer(state_dict):
#     """Yields a RecordBatch for each parameter in the model state dict"""
#     for param_name, param in state_dict.items():
#         if not isinstance(param, torch.Tensor):
#             continue
#         param_shape = list(param.size())
#         param_value = param.flatten().tolist()
#         yield pa.RecordBatch.from_arrays(
#             [
#                 pa.array(
#                     [param_name],
#                     pa.string(),
#                 ),
#                 pa.array(
#                     [param_value],
#                     pa.list_(pa.float64(), -1),
#                 ),
#                 pa.array(
#                     [param_shape],
#                     pa.list_(pa.int64(), -1),
#                 ),
#             ],
#             ["name", "value", "shape"],
#         )

# 递归展开嵌套 dict，支持 model + optimizer 状态
def _save_model_writer(state_dict):
    """递归展开嵌套 dict，支持 model + optimizer 状态"""
    def recur(prefix, obj):
        if isinstance(obj, torch.Tensor):
            # 到达底层张量，输出一行
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array([prefix], pa.string()),
                    pa.array([obj.flatten().tolist()], pa.list_(pa.float64(), -1)),
                    pa.array([list(obj.size())], pa.list_(pa.int64(), -1)),
                ],
                ["name", "value", "shape"],
            )
        elif isinstance(obj, dict):
            # 继续向下拆
            for k, v in obj.items():
                yield from recur(f"{prefix}.{k}" if prefix else k, v)
        # 其他类型（int/float/list 等）直接忽略
        else:
            return

    # 从顶层开始递归
    yield from recur("", state_dict)

# 保存模型，包括版本控制
def save_model(state_dict: OrderedDict, file_name: str, version=False):
    """Saves a PyTorch model in lance file format

    Args:
        state_dict (OrderedDict): Model state dict
        file_name (str): Lance model name
        version (bool): Whether to save as a new version or overwrite the existing versions,
            if the lance file already exists
    """
    # Create a reader
    reader = pa.RecordBatchReader.from_batches(
        GLOBAL_SCHEMA, _save_model_writer(state_dict)
    )

    if os.path.exists(file_name):
        if version:
            # If we want versioning, we use the overwrite mode to create a new version
            lance.write_dataset(
                reader, file_name, schema=GLOBAL_SCHEMA, mode="overwrite"
            )
        else:
            # If we don't want versioning, we delete the existing file and write a new one
            shutil.rmtree(file_name)
            lance.write_dataset(reader, file_name, schema=GLOBAL_SCHEMA)
    else:
        # If the file doesn't exist, we write a new one
        lance.write_dataset(reader, file_name, schema=GLOBAL_SCHEMA)

# 加载模型
def _load_weight(weight: dict) -> torch.Tensor:
    """Converts a weight dict to a torch tensor"""
    return torch.tensor(weight["value"], dtype=torch.float32).reshape(weight["shape"])

def _load_state_dict(file_name: str, version: int = 1, map_location=None) -> OrderedDict:
    """Reads the model weights from lance file and returns a model state dict
    If the model weights are too large, this function will fail with a memory error.

    Args:
        file_name (str): Lance model name
        version (int): Version of the model to load
        map_location (str): Device to load the model on

    Returns:
        OrderedDict: Model state dict
    """
    ds = lance.dataset(file_name, version=version)
    weights = ds.take([x for x in range(ds.count_rows())]).to_pylist()
    state_dict = OrderedDict()

    for weight in weights:
        state_dict[weight["name"]] = _load_weight(weight).to(map_location)

    return state_dict

def load_model(
    model: torch.nn.Module, file_name: str, version: int = 1, map_location=None
):
    """Loads the model weights from lance file and sets them to the model

    Args:
        model (torch.nn.Module): PyTorch model
        file_name (str): Lance model name
        version (int): Version of the model to load
        map_location (str): Device to load the model on
    """
    state_dict = _load_state_dict(file_name, version=version, map_location=map_location)
    model.load_state_dict(state_dict)


def test_save_model():
    # 1. 加载 .pt 文件得到 state_dict
    ckpt = torch.load("checkpoints/clip_flickr8k/clip_flickr8k_ep5.pt", map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt   # 兼容不同保存格式


    # 展平：把 7 个子 dict 合并成一级 dict
    flat_state = OrderedDict()
    for tower_name, sub_dict in [
        ("img_encoder",  ckpt["img_encoder"]),
        ("img_head",     ckpt["img_head"]),
        ("text_encoder", ckpt["text_encoder"]),
        ("text_head",    ckpt["text_head"]),
        ("optimizer",    ckpt["optimizer"]),      # 优化器状态也存
    ]:
        for param_name, param in sub_dict.items():
            flat_state[f"{tower_name}.{param_name}"] = param

    # 标量转成 0-D 张量，统一接口
    flat_state["epoch"] = torch.tensor(ckpt["epoch"])
    flat_state["loss"]  = torch.tensor(ckpt["loss"])

    # print(flat_state.keys())
    # 2. 写入 Lance（支持版本）
    save_model(
        state_dict=flat_state,
        file_name="checkpoints/clip_flickr8k_lance",  # 目录名
        version=True                                  # True=生成新版本，False=覆盖
    )
    print("✅ 已转换为 Lance 格式")

def test_load_model():
    from train_clip_flickr8k import ImageEncoder, TextEncoder, Head

    # 1. 重新构造网络（与训练时结构完全一致）
    img_encoder = ImageEncoder("resnet50").to("cuda")
    img_head    = Head(2048, 256).to("cuda")
    text_encoder= TextEncoder("bert-base-cased").to("cuda")
    text_head   = Head(768, 256).to("cuda")

    # 2. 统一加载扁平权重
    lance_path = "checkpoints/clip_flickr8k_lance"
    full_dict  = _load_state_dict(lance_path, version=1, map_location="cuda")

    # 3. 按前缀拆回各个子模块
    img_encoder.load_state_dict(
        OrderedDict((k[len("img_encoder."):], v) for k, v in full_dict.items() if k.startswith("img_encoder.")))
    img_head.load_state_dict(
        OrderedDict((k[len("img_head."):], v) for k, v in full_dict.items() if k.startswith("img_head.")))
    text_encoder.load_state_dict(
        OrderedDict((k[len("text_encoder."):], v) for k, v in full_dict.items() if k.startswith("text_encoder.")))
    text_head.load_state_dict(
        OrderedDict((k[len("text_head."):], v) for k, v in full_dict.items() if k.startswith("text_head.")))

    print("✅ 所有子模块已恢复，可直接推理或继续训练！")

if __name__ == "__main__":
    ds = lance.dataset("checkpoints/clip_flickr8k_lance")   # 替换成你的路径
    print(ds.versions())                     # 返回列表，每项含 id / timestamp / 