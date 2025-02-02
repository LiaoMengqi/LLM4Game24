*最新消息*
- [2025/02/02] 上传 DeepSeek R1 Zero 强化学习 pipeline 代码与实验报告，包含 GRPO 和基于规则的奖励模型的实现。
- [2025/01/29] 优化案例生成与微调数据生成算法效率，训练数据的输入数字也会随机打乱，增强数据多样性。更换 system prompt， 用于之后强化学习的训练。
新增混合数据微调，更新新版微调实验报告。
- [2025/01/26] 上传 SFT 代码与实验报告。

## 简介


这个仓库是在 *解决 24 点游戏* 的背景下的大模型 Long CoT 微调与强化学习的 toy project，
旨在为大家提供大模型 Long CoT 微调与强化学习的入门实践。

**24 点游戏简介**：  
24 点游戏的目标是给定四个 1-13 的数字，通过四则运算（加、减、乘、除）的组合，使得计算结果等于 24。例如，给定数字 `4, 2, 6, 3`，可以通过 `(4 × 3) + (2 × 6)` 得到结果 24。

**项目内容**：  
- 提供了 Long CoT 推理数据的生成代码，以及 Qwen 模型的微调和推理代码。  
- 我们实现了 DeepSeek R1-Zero 的强化学习 pipeline，以进一步提升模型在 24 点游戏中的表现能力。
- 数据格式设计以及实验结果详见实验报告 [`report/sft_report_zh.pdf`](report/sft_report_zh.pdf)。  
- 强化学习部分的实验结果详见实验报告 [`report/rl_report_zh.pdf`](report/rl_report_zh.pdf)。  

此项目仅供学习和参考之用，欢迎大家关注后续更新！

**未来计划**：
- 实现 DPO 进行偏好学习。 

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
评估命令，生成推理步骤与结果，生成结果保存在 `./result` 目录下
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--path /root/autodl-tmp/ \
--checkpoint_id short_v3/checkpoint-900 \
--save_file short_v3.json \
--batch_size 16 \
--max_length 4096
```

`result_statistic.ipynb` 中的代码根据生成的结果指标计算和绘图。

## 强化学习

GRPO 强化学习用到了两个设备用于放 actor 模型和 reference 模型，
如果只有一张卡可以设置 `CUDA_VISIBLE_DEVICES=0`,并设置 `--ref_device 0`。
```shell
CUDA_VISIBLE_DEVICES=5,6 python rl.py --lr 2e-6 \
--beta 0.01 \
--max_momentum 0.2 \
--save_dir mt02 \
--max_step 50 \
--save_step 10 \
--batch_size 12 \
--group_size 8 \
--mini_batch_size 4 \
--lambda_oc 0.55 \
--lambda_pr 0.3 \
--lambda_l 0.15 \
--top_p 0.9 \
--temperature 0.7
```

参数说明：
- `--beta` KL 惩罚系数。
- `--max_momentum` 用于设置奖励的动态归一化，默认为 0，表示不使用动态归一化，DeepSeek Math 中没有使用动态归一化。
- `--batch_size` 是每次迭代用到的实例数量。
- `--group_size` 是 GRPO 的组大小，即对于每个实例生成的样本数量。
- `--mini_batch_size` 是进行策略更新时的批量大小，影响显存占用。
- `--lambda_oc` 奖励模型对于结果奖励的权重。
- `--lambda_pr` 奖励模型对于过程正确性奖励的权重。
- `--lambda_l` 奖励模型对于长度奖励的权重。
- `--top_p` 是对生成的候选答案进行采样的 top_p 值，默认为 0.9。
- `--temperature` 是对生成的候选答案进行采样的 temperature 值，默认为 0.7。