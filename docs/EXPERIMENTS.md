# 实验范围与来源

| 内容 | 来源 | 展示范围 |
|---|---|---|
| SDXL OBS10–60 LPIPS 图 | 项目成员提供的汇总图 | 同已有评测集上的结果，训练配置并非全部一致 |
| SDXL OBS30 定性图 | 项目成员提供的论文图 | 四个提示词，不能代替整体质量评价 |
| SD2.1 OBS30 | `elsarticle_paper (4).pdf` 第32页，Figure A.10 | 论文附录的补充定性结果 |
| SD1.5 OBS30 | 同一 PDF 第35页，Figure A.13 | 论文附录的补充定性结果 |
| 普通续训对照 | `results/continuation_comparison.csv` | 100张已有图片的汇总，参见 README 的限制 |

SD1.5 和 SD2.1 均按论文图的原列顺序展示：Full、Pruned + Skip、Repair LoRA + Skip、Ours Stage2 + Skip。图像从 PDF 内嵌图直接提取；没有通过生成模型重绘或修饰实验输出。页面引用用于追溯来源，不代表该 PDF 已通过同行评审。

跨模型图只支持初步定性观察；未在本仓库提供对应完整训练资产和独立定量指标。不据此宣称方法对所有骨干、提示词或指标都有效。

SDXL 主要设置：35步、DDIM、CFG 5.0、1024×1024评测、threshold 0.014、max_skip_steps 4；Stage2 常用配置为 rank8、500步、lr5e-5、512×512、训练threshold 0.010、latent-only。实际各比例 prompt 数及初始化来源存在差异，不能将常用参数视为所有存档权重的统一配置。

LPIPS / PSNR 测量与对应 Full 参考图的差异，CLIP 关注文本一致性，DINO 关注特征距离；均不能单独代表主观画质。耗时是历史设备记录，新的硬件、环境和实现可能不同。有关训练评测 prompt 重叠和对照不完全匹配的限制保留在 README。

论文的 arXiv 链接将在正式公开并获得编号后补充；目前不提供占位编号或声称已经接收。
