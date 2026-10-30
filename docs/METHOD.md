# 方法与代码对应

## 研究问题

剪枝改变去噪网络的近似误差；自适应跳步则在部分时刻复用过去的噪声预测。我们研究如何让已经修复的剪枝模型适应这种预测复用，而不是仅仅把两个加速模块串联。

## 两阶段流程

Stage1 修复剪枝学生；Stage2 从该 LoRA 初始化，固定完整教师和学生基础权重，只更新学生 LoRA。两阶段不是同时运行两个教师，也不是在推理时额外调用完整模型。

对一个记录下来的跳步片段，教师与学生使用相同 prompt 和起点潜变量。教师逐步重新预测噪声；学生在片段锚点预测一次噪声，随后复用该预测，通过同一个调度器展开片段内的更新。损失匹配片段终点潜变量，不是 RGB 图像，也不是整条采样轨迹的最终图像。

```math
\mathcal L_{\mathrm{latent}}=\operatorname{MSE}(z^{\mathrm{student}}_{\mathrm{end}},\operatorname{stopgrad}(z^{\mathrm{teacher}}_{\mathrm{end}})).
```

最终 latent-only 实验禁用锚点 epsilon 正则项。历史代码仍保留该项以及其他实验选项，默认参数不等同于最终实验配置。

## 代码入口

- `scripts/train_skip_aware_lora_sdxl.py`：采集片段、展开局部轨迹、计算 `segment_loss`、只更新 LoRA。
- `adaptive_diffusion/acceleration/sparse_pipeline.py`：修改版自适应跳步推理实现，基于上游项目。
- `scripts/obs_pruned_skip_only.py`：加载剪枝模型和 LoRA、生成图片并记录耗时与跳步率。
- `scripts/compute_paper_metrics.py`：使用对应完整模型图片作为配对参考。

## 复现范围

当前提供 SDXL 核心续训、推理和指标代码；不包含底模、剪枝 UNet、Stage1 数据集和全部训练权重。SD1.5 / SD2.1 图片来自论文附录，不意味着本目录中的 SDXL 脚本直接支持这些模型。完整流程尚未在新环境端到端复验。

流程图由图像生成工具辅助绘制；可编辑的结构简图为 `results/method_overview.drawio`。图像示意中的装饰性潜变量和输出缩略图不是实验结果。
