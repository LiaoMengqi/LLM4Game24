import json
import random

from tqdm import tqdm

from utils import *


def format_data(data):
    dataset = []
    for item in data:
        temp = dict()
        item = item.split('\n')
        temp["input"] = item[0] + '\n'
        temp["output"] = "\n".join(item[1:])
        dataset.append(temp)
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--regen_case", action="store_true", default=False)
    args = parser.parse_args()

    if args.regen_case:
        print("collect all valid combinations ...")
        cases = generate_24_game_combinations()
        # shuffle
        random.shuffle(cases)
        test_cases = cases[:100]
        train_cases = cases[100:]

        with open("./dataset/train_cases.json", "w") as f:
            json.dump(train_cases, f)

        with open("./dataset/test_cases.json", "w") as f:
            for i in range(len(test_cases)):
                # shuffle test case
                random.shuffle(test_cases[i])
            json.dump(test_cases, f)
    else:
        with open("./dataset/train_cases.json", "r") as f:
            train_cases = json.load(f)
        # with open("./dataset/test_cases.json", "r") as f:
        #     test_cases = json.load(f)

    print("collect sft data ...")
    max_steps_set = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    short_datasets = []
    medium_datasets = []
    long_datasets = []

    datasets = []
    sample_size = 3
    tqdm_bar = tqdm(total=len(train_cases))
    for case in train_cases:
        full_reasoning_paths = []
        for i in range(sample_size):
            # 随机搜索
            random.shuffle(case)
            reasoning_path = solve_24_with_steps(case)
            if reasoning_path is not None:
                full_reasoning_paths.append(reasoning_path)

        # compress
        for i in range(len(max_steps_set)):
            temp_set = set()
            for case_res in full_reasoning_paths:
                compressed_log = compress_search_logs(
                    logs=case_res,
                    max_lines=max_steps_set[i]
                )
                temp_set.add("\n".join(compressed_log))
            datasets += list(temp_set)
        tqdm_bar.update(1)

    with open("dataset/train/train.json", "w") as f:
        datasets = format_data(datasets)
        json.dump(datasets, f)

