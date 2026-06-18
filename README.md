# Vision-Language Clustering (VLC)

**当前主线：** fixed-K **VLC-policy**（包名 `vlc`）— 给定 K 与指令，序贯动作 `ASSIGN` / `MOVE` / `NO_OP` / `STOP` 完成聚类。

批量 **`ClusterDecoder`**（`vlc/model.py`）为 **baseline**，非主方法。

---

## 目录结构（分子包）

```
vlc/
  policy/       # 主线 fixed-K 策略
  baseline/     # 批量 ClusterDecoder + 经典 baselines
  data/         # 特征、编码器、伪标签
  core/         # utils、metrics、losses
  bench/        # same-K、dsprites、多指令、约束
  _legacy.py

scripts/lite/
configs/lite/
legacy/
```

详见 `vlc/README.md`。旧写法 `from vlc.features import ...` 仍可用（自动映射到 `vlc.data.features`）。

---

## Backbone

| Backbone | 状态 |
|----------|------|
| `SmallBackbone` | 快速对照 |
| `LLMBackbone` | CIFAR-10 fixed-K 已跑通 |
| `VLMBackbone` | 已支持；全量评估待定 |

---

## 常用命令

```bash
python -m vlc train --config configs/lite/cifar10_fixed_k_llm.yaml
python -m vlc same-k --config configs/lite/dsprites_same_k.yaml --prepare
python scripts/check_imports.py
```

包根入口：`vlc/main.py`（`python -m vlc <子命令>`）。脚本与模块对应见 [scripts/lite/README.md](scripts/lite/README.md)。

任务规模超参：**`episode_size`**（一次聚类图像数）。**`batch_size`** 仅用于 batch decoder。

---

## 导入

```python
from vlc.policy import ClusteringActionPolicy, build_backbone, train_clustering_policy
from vlc.baseline.model import ClusterDecoder
from vlc.core.utils import load_config
from vlc.data.features import task_dir
```

历史代码见 `legacy/README.md`（原 `vlc_lite`、Full `vlc_gen` 自回归等）。
