## 简介

这个 repo 是一个微调大模型解决 24 点游戏的 toy project ，仅供大家了解大模型长 CoT 推理微调。
我们还实现了 Deep seek R1-Zero 的强化学习 pipeline，进一步提高了模型的能力。

24 点游戏是指给定四个 1-13 的数字，找到一种四则基本运算（加减乘除）的组合能将四个数字的运算结果达到24。
比如给定数字 4 2 6 3，可以通过 (4 x 3) + (2 x 6) 得到 24。

我们提供了长 CoT 推理数据生成代码以及 Qwen 的微调和推理代码。
数据格式设计以及实验结果可以查看实验报告 `report/sft_report.pdf`。
强化学习部分的实验报告可以查看 `report/rl_report.pdf`。


## 微调数据数据生成

我们已经生成数据，这部分可以跳过，生成的数据存放在`dataset/train`目录下，`train.json`包含所有训练数据，
`short.json, medium.json, long.json`分别包含短、中、长推理路径的数据。 
如果需要重新生成数据可以运行以下命令：

```shell
python gen_sft_data.py
```
消融实验数据生成：

```shell
python gen_abation_data.py
```

## 微调与评估

微调命令
```shell
python stf.py \
--dataset ./dataset/short_data.json \
--output_dir short_v3 \
--max_steps 1500 \
--batch_size 8 \
--accumulation_steps 4 \
--max_len 1400
```
评估命令，进行生成，生成结果保存在 result 目录下
```shell
python evaluate.py \
--path /root/autodl-tmp/ \
--checkpoint_id short_v3/checkpoint-900 \
--save_file short_v3.json \
--max_length 4096
```

指标计算与绘图：运行 result_statistic.ipynb 中所有命令