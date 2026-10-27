# 使用说明与限制

这是从实验脚本整理出的展示候选版，不是经过完整环境验证的一键复现项目。需要自行准备 SDXL 模型、与其匹配的已剪枝 UNet、对应结构的 Stage1 LoRA、训练与评测 prompt，以及兼容的 GPU 软件环境。不同结构的剪枝模型不能直接复用同一 LoRA。

目录中包含：

- `scripts/train_skip_aware_lora_sdxl.py`：跳步感知续训练，包含其他历史实验选项。
- `scripts/obs_pruned_skip_only.py`：加载剪枝模型和可选 LoRA 后进行跳步推理。
- `scripts/compute_paper_metrics.py`：相对 Full SDXL 参考图计算配对指标。
- `adaptive_diffusion/`：实验使用的修改版 AdaptiveDiffusion 跳步实现及上游许可证。
- `results/continuation_comparison.csv`：已有的续训对比汇总，不包含原始图片。

## 原始环境快照

服务器备份中找到了原始环境快照：[Conda 环境](environment/conda_environment_snapshot.yml)（已去除机器专属 prefix）和 [pip 软件包清单](environment/pip_freeze_snapshot.txt)。它们是历史环境记录，不是经过新机器验证的跨平台安装锁文件，其中可能包含无法直接解析的本地包或旧版本。主要版本如下：

| 软件 | 备份记录 |
|---|---|
| Python | 3.10.20 |
| PyTorch / torchvision | 2.7.1+cu118 / 0.22.1+cu118 |
| diffusers / transformers | 0.31.0 / 4.45.0 |
| PEFT / accelerate | 0.19.1 / 1.12.0 |
| numpy / Pillow / LPIPS | 1.26.4 / 12.0.0 / 0.1.4 |

完整评测还需要 CLIP / DINO 对应模型文件。软件包清单内的 `cuda-toolkit` 版本不等同于 PyTorch 构建的 CUDA 版本；驱动、实际 GPU、加载的 CUDA 库及依赖冲突均需要单独检查。当前候选版没有在全新环境进行端到端安装与实验验证。

训练和推理脚本的 `--model-path`、`--pruned-path` 和 `--output-dir` 必须自行指定。Stage2 的 latent-only 实验需显式指定 `--loss-mode unrolled --unrolled-loss-mode final --anchor-eps-weight 0`，不能直接依赖默认参数。使用 `--help` 查看其余选项；模型和数据齐备之前，不建议直接运行耗时训练。Stage1 图像微调和 OBS 剪枝的完整上游项目及其数据准备流程尚未打包，需要按原项目说明自行准备；本仓库不是从零开始的一键复现包。

指标脚本要求参考图、方法图和 prompt 数量一致且图像文件名编号匹配。它无法单凭文件名验证 prompt 和随机种子是否真实对应，需检查生成记录。默认在 512×512 上计算配对指标，即使原图是 1024×1024。

**安全提示**：脚本读取的 `.pkl` 和 PyTorch 模型可能触发任意代码执行，只能加载可信来源的文件。不要把密码、服务器地址、未经授权的模型或数据上传到公开仓库。
