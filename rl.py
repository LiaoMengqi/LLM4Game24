import json
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed as transformers_seed

from grpo import (GRPOTrainer4Game24, Dataset4Game24, RewardModel4Game24)


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def custom_collate_fn(batch):
    # 假设批量数据是一个列表，这里将每个元素平方
    prompt_template = '<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'
    system_prompt = ("Play 24 game. "
                     "Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations."
                     " Output the reasoning steps, one step per line. On the last line, output the expression, for example, "
                     "input: '10 1 12 3', "
                     "the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.")
    model_inputs = {'prompt': [], 'case': []}
    model_inputs['prompts'] = [prompt_template.format(system_prompt=system_prompt, user_prompt=x['prompt']) for x in
                               batch]
    model_inputs['cases'] = [x['case'] for x in batch]
    return model_inputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--checkpoint", type=str,
                        default='./checkpoint/format_v3/checkpoint-1000/')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dataset", type=str, default='./dataset/train_cases.json')
    parser.add_argument("--grpo_iterations", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=5)
    parser.add_argument("--lambda_oc", type=float, default=0.15)
    parser.add_argument("--lambda_pr", type=float, default=0.75)
    parser.add_argument("--lambda_l", type=float, default=0.1)
    parser.add_argument("--act_device", type=int, default=0)
    parser.add_argument("--ref_device", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=12)  # number of instance per batch
    parser.add_argument("--group_size", type=int, default=10)  # number of samples for each instance
    parser.add_argument("--mini_batch_size", type=int, default=5)  # batch size to update policy
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--max_step", type=int, default=200)
    parser.add_argument("--save_step", type=int, default=40)
    parser.add_argument("--save_dir", type=str, default='rl')
    parser.add_argument("--max_momentum", type=float, default=0.99)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.3)
    parser.add_argument("--final_temperature", type=float, default=0.9)

    args = parser.parse_args()

    set_seed(42)

    # dataset
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    dataset = Dataset4Game24(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    reward_model = RewardModel4Game24(lambda_l=args.lambda_l, lambda_pr=args.lambda_pr, lambda_oc=args.lambda_oc)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.padding_side = "left"

    trainer = GRPOTrainer4Game24(
        policy_model=model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        args=args
    )
    loss_list, acc_list, sentence = trainer.train(dataloader)
    # print(loss_list)
    import json

    with open(f'./results/rl_{args.save_dir}_info.json', 'w') as f:
        json.dump({'loss_list': loss_list, 'acc_list': acc_list, 'samples': sentence}, f)
