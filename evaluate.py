import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def process_data():
    prompt_template = '<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'
    system_prompt=("Play 24 game. "
                   "Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations."
                   " Output the reasoning steps, one step per line. On the last line, output the expression, for example, "
                   "input: '10 1 12 3', "
                   "the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.")
    with open(args.data, "r") as f:
        data = json.load(f)
    inputs = []
    for case in data:
        case = [str(e) for e in case]
        inputs.append(prompt_template.format(system_prompt=system_prompt, user_prompt=" ".join(case) + '/n'))
    return inputs


def batch_generate(model, tokenizer, input_texts, batch_size=8):
    all_results = []
    tqdm_bar = tqdm(total=len(input_texts) // batch_size + (len(input_texts) % batch_size != 0), desc="Generating")

    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:min(i + batch_size, len(input_texts))]
        # 对batch内的文本进行填充和截断
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        # print(batch_texts)
        # 生成文本

        with torch.no_grad():
            pass
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=args.max_length,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            # 解码生成的文本
        output_ids = output_ids.cpu()
        generated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_results.extend(generated_texts)
        tqdm_bar.update(1)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/root/autodl-tmp/")
    parser.add_argument("--checkpoint_id", type=str, default="short/checkpoint-800")
    parser.add_argument("--data", type=str, default="./dataset/test_cases.json")
    parser.add_argument("--save_file", type=str, default="short.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    test_datasets = process_data()

    # 加载检查点
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint_dir = args.path + args.checkpoint_id
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.padding_side = "left"

    # 设置为评估模式
    model.eval()

    all_results = batch_generate(
        model,
        tokenizer,
        test_datasets,
        batch_size=args.batch_size
    )
    # 使用tqdm显示进度
    # num_batches = (len(test_datasets) + args.batch_size - 1) // args.batch_size
    # with tqdm(total=num_batches, desc="Generating") as pbar:

    # pbar.update(1)

    # 保存结果
    with open("results/" + args.save_file, "w") as f:
        json.dump(all_results, f)
