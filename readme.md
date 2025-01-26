# 微调数据数据生成

生成的数据存放在`data`文件夹下。

如果需要重新生成数据可以依次运行以下命令

dataset 目录下运行

```shell
python gen_case.py
python split_cases.py
```
根目录运行（时间较久）
```shell
python gen_data.py
```

dataset 目录运行
```shell
python trans_format.py
```

执行根录下 data_statistic.ipynb 中所有命令进行数据集的划分

根目录下运行以下命令生成消融实验数据
```shell
python gen_abation_data.py
```

# 微调与评估

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

指标计算与绘图，运行 result_statistic.ipynb 中所有命令