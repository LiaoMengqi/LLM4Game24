*最新消息*
- [2025/02/02] 上传 DeepSeek R1 Zero 强化学习 pipeline 代码，包含 GRPO 和基于规则的奖励模型的实现。
- [2025/01/29] 优化案例生成与微调数据生成算法效率，训练数据的输入数字也会随机打乱，增强数据多样性。更换 system prompt， 用于之后强化学习的训练。
新增混合数据训练，更新新版微调实验报告。
- [2025/01/26] 上传 SFT 代码与实验报告。

## 简介


这个仓库是在 *解决 24 点游戏* 的背景下的大模型 Long CoT 微调与强化学习的 toy project，
旨在帮助大家了解大模型在长链推理（CoT）场景下的微调方法与强化学习的实践。

**24 点游戏简介**：  
24 点游戏的目标是给定四个 1-13 的数字，通过四则运算（加、减、乘、除）的组合，使得计算结果等于 24。例如，给定数字 `4, 2, 6, 3`，可以通过 `(4 × 3) + (2 × 6)` 得到结果 24。

**项目内容**：  
- 提供了 Long CoT 推理数据的生成代码，以及 Qwen 模型的微调和推理代码。  
- 我们实现了 DeepSeek R1-Zero 的强化学习 pipeline，以进一步提升模型在 24 点游戏中的表现能力。
- 数据格式设计以及实验结果详见实验报告 [`report/sft_report.pdf`](report/sft_report.pdf)。  
- 强化学习部分的实验结果详见实验报告 [`report/rl_report.pdf`](report/rl_report.pdf)。  

此项目仅供学习和参考之用，欢迎大家关注后续更新！


## 微调数据数据生成

我们已经生成数据，这部分可以跳过，生成的数据存放在`dataset/train`目录下，`train.json`包含所有训练数据，
`short.json, medium.json, long.json`分别包含短、中、长推理路径的数据。 
如果需要重新生成数据可以运行以下命令：

```shell
python gen_sft_data.py
```
不同格式实验数据生成：

```shell
python gen_abation_data.py
```

## 微调与评估

微调命令
```shell
CUDA_VISIBLE_DEVICES=0 python stf.py \
--dataset ./dataset/train/short.json \
--output_dir short_v3 \
--max_steps 1000 \
--batch_size 8 \
--accumulation_steps 4 \
--max_len 1250
```
评估命令，进行生成，生成结果保存在 result 目录下
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--path /root/autodl-tmp/ \
--checkpoint_id short_v3/checkpoint-900 \
--save_file short_v3.json \
--batch_size 16 \
--max_length 4096
```

指标计算与绘图：运行 result_statistic.ipynb 中所有命令

## 强化学习
