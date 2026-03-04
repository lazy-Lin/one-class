# 单分类图像异常检测项目

基于 `PyTorch + timm + FAISS` 的单分类图像检测工程，支持：
- 正样本训练并自动标定阈值
- 单张图像离线推理
- ONNX 导出与 ONNX Runtime 推理
- 透传调用 anomalib 命令

## 1. 环境准备

推荐 Python `3.10+`。

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. 项目结构

```text
scripts/
  config.py         # 训练参数配置（train.py 从这里读取）
  train.py          # 训练入口
  predict.py        # PyTorch 推理
  export_onnx.py    # 导出 ONNX
  infer_onnx.py     # ONNX Runtime 推理
  run_anomalib.py   # 透传 anomalib 命令
src/one_class/
  data.py
  model.py
  pipeline.py
```

## 3. 数据组织

`train.py` 使用 `scripts/config.py` 中的路径参数。默认目录如下：

```text
data/
  train/
    normal/
      *.jpg / *.png ...
  val/
    normal/
      *.jpg / *.png ...
```

说明：
- `train_dir`：用于建立正常样本特征库（FAISS 索引）
- `val_dir`：用于计算分数分布并按分位数确定阈值

## 4. 训练（参数在 config.py）

编辑 `scripts/config.py` 的 `CONFIG`：

```python
CONFIG = TrainConfig(
    train_dir="data/train/normal",
    val_dir="data/val/normal",
    output_dir="artifacts/faiss_patchcore",
    backbone="resnet18",
    image_size=224,
    batch_size=64,
    num_workers=4,
    knn_k=5,
    threshold_quantile=0.995,
    device=None,
)
```

然后直接启动训练：

```bash
python scripts/train.py
```

训练完成后会在 `output_dir` 生成：
- `faiss.index`
- `metadata.json`

## 5. 推理（PyTorch）

```bash
python scripts/predict.py --artifact-dir artifacts/faiss_patchcore --image path/to/test.jpg
```

输出字段：
- `prediction`：`normal` 或 `anomaly`
- `score`：当前样本异常分数
- `threshold`：训练阶段标定阈值

## 6. 导出与推理（ONNX）

导出：

```bash
python scripts/export_onnx.py --artifact-dir artifacts/faiss_patchcore --output artifacts/model.onnx
```

ONNX Runtime 推理：

```bash
python scripts/infer_onnx.py --artifact-dir artifacts/faiss_patchcore --onnx artifacts/model.onnx --image path/to/test.jpg
```

## 7. anomalib 透传

可通过以下方式透传 anomalib 命令参数：

```bash
python scripts/run_anomalib.py <anomalib原生命令参数>
```

示例：

```bash
python scripts/run_anomalib.py --help
```

## 8. 参数建议

- `backbone`：先从 `resnet18` 起步，资源充足可尝试更大模型
- `threshold_quantile`：常见范围 `0.99 ~ 0.999`
- `knn_k`：通常 `3 ~ 10` 可作为起点
- `batch_size`：按显存逐步上调

## 9. 常见问题

- 训练报错 “train_dir 下没有可用图像”：
  - 检查 `scripts/config.py` 的目录是否正确
  - 支持后缀：`.jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp`
- 推理分数波动大：
  - 增加验证集规模
  - 调整 `threshold_quantile` 与 `knn_k`
