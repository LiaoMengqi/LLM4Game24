import json

from tqdm import tqdm

from dataset import *

with open("./dataset/train_cases.json", "r") as f:
    train_cases = json.load(f)

max_lines_set = [4, 8, 12, 16, 20]
datasets = []
sample_size = 5

tqdm_bar = tqdm(total=len(train_cases))
for case in train_cases:
    sampled_case_res = []
    for i in range(sample_size):
        # 随机搜索
        result = solve_24_with_steps(case)
        if result is not None:
            sampled_case_res.append(result)

    # compress
    for i in range(len(max_lines_set)):
        temp_set = set()
        for case_res in sampled_case_res:
            compressed_log = compress_search_logs(
                logs=case_res,
                max_lines=max_lines_set[i]
            )
            temp_set.add("\n".join(compressed_log))
        datasets += list(temp_set)
    tqdm_bar.update(1)

with open("dataset/train_data.json", "w") as f:
    json.dump(datasets, f)
