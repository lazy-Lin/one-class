# 单分类图像异常检测项目

基于 `PyTorch + timm + FAISS` 的单分类图像检测工程，支持：
- 正样本训练并自动标定阈值
- 单张图像离线推理
- ONNX 导出与 ONNX Runtime 推理
- 透传调用 anomalib 命令

## 项目基本思想

这个项目实现的是“只学习正常样本”的单分类检测。核心逻辑不是去学习“正常 vs 异常”的二分类边界，而是先学习“什么是正常分布”，再用距离或相似度判断新样本是否偏离该分布。

### 为什么可以实现单分类

1. 预训练视觉骨干网络（如 `resnet18`）能够把图像映射到语义特征空间。  
2. 大量正常样本在该特征空间会形成相对稳定的“正常区域”。  
3. 新样本如果属于正常类，其特征通常靠近正常区域；异常样本则更容易远离该区域。  
4. 通过验证集统计分数分布并设定阈值，就能完成“正常/异常”判定，而无需显式负样本。

### 本项目里的具体实现

- 用 `timm` 骨干提取图像 embedding。  
- 用 `FAISS` 构建正常样本特征索引。  
- 用 kNN 距离均值作为异常分数（`score`）。  
- 用验证集分位数（`threshold_quantile`）自动得到阈值（`threshold`）。  
- 推理时只要 `score <= threshold` 判为正常，否则判为异常。

### 适用前提

- 正样本数量足够且覆盖真实业务波动（光照、角度、背景、设备差异）。  
- 验证集与线上分布接近，用于稳定阈值标定。  
- 异常与正常在特征空间存在可分离性。

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

## 2.1 脚本功能解释

- `scripts/config.py`
  - 作用：集中管理训练参数，`train.py` 从这里读取配置
  - 主要内容：`TrainConfig` 与全局 `CONFIG`
  - 你通常会改这里：数据路径、backbone、batch size、阈值分位数

- `scripts/train.py`
  - 作用：训练入口，执行单分类训练流程
  - 输入：`config.py` 中的 `CONFIG`
  - 输出：`output_dir/faiss.index` 与 `output_dir/metadata.json`

- `scripts/predict.py`
  - 作用：基于 PyTorch 对单张图片做离线推理
  - 输入：`--artifact-dir`（训练产物目录）、`--image`（待测图片）
  - 输出：`prediction`、`score`、`threshold`

- `scripts/export_onnx.py`
  - 作用：导出特征提取模型为 ONNX
  - 输入：`--artifact-dir`（读取训练时的 backbone 和 image_size）
  - 输出：ONNX 模型文件（默认 `artifacts/model.onnx`）

- `scripts/infer_onnx.py`
  - 作用：使用 ONNX Runtime 进行推理，并复用 FAISS 阈值判定
  - 输入：`--artifact-dir`、`--onnx`、`--image`
  - 输出：`prediction`、`score`、`threshold`

- `scripts/run_anomalib.py`
  - 作用：透传调用 anomalib CLI，便于直接使用官方命令
  - 输入：anomalib 原生命令参数
  - 输出：等同于 anomalib 命令行输出

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

## 3.1 全流程介绍（数据准备 → 训练 → 推理 → 部署）

### Step 1：准备数据

1. 收集正常样本图像，并按场景尽量覆盖完整分布（光照、角度、背景、设备差异）。
2. 划分训练集和验证集，建议同分布划分，且验证集尽量独立。
3. 按以下目录放置：

```text
data/
  train/
    normal/
      xxx.jpg
  val/
    normal/
      yyy.jpg
```

### Step 2：配置训练参数

在 `scripts/config.py` 中配置：
- 数据路径：`train_dir`、`val_dir`
- 模型参数：`backbone`、`image_size`
- 训练性能参数：`batch_size`、`num_workers`、`device`
- 检测策略参数：`knn_k`、`threshold_quantile`

实践建议：
- 显存有限时优先降低 `batch_size`
- 误报较多时可适当提高 `threshold_quantile`
- 漏报较多时可适当降低 `threshold_quantile`

### Step 3：执行训练

```bash
python scripts/train.py
```

训练阶段会完成：
1. 用 backbone 提取训练集特征
2. 建立 FAISS 索引（正常样本特征库）
3. 用验证集计算分数分布并确定阈值
4. 保存产物到 `output_dir`

核心产物：
- `faiss.index`：正常样本特征索引
- `metadata.json`：训练配置与阈值等元信息

### Step 4：离线推理验证（PyTorch）

```bash
python scripts/predict.py --artifact-dir artifacts/faiss_patchcore --image path/to/test.jpg
```

输出说明：
- `prediction=normal`：分数低于阈值
- `prediction=anomaly`：分数高于阈值
- `score` 越大表示越偏离正常分布

### Step 5：导出 ONNX 并验证推理

```bash
python scripts/export_onnx.py --artifact-dir artifacts/faiss_patchcore --output artifacts/model.onnx
python scripts/infer_onnx.py --artifact-dir artifacts/faiss_patchcore --onnx artifacts/model.onnx --image path/to/test.jpg
```

这一步用于验证部署链路，确保导出模型与训练阈值策略一致。

### Step 6：上线前检查清单

1. 用一批真实线上样本做抽检，确认阈值满足业务目标。
2. 对误报样本做回看，评估是否需要扩充训练数据覆盖范围。
3. 固化版本：保存当前 `faiss.index`、`metadata.json`、`model.onnx`。
4. 约定重训周期，数据分布变化时更新阈值和索引。

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
