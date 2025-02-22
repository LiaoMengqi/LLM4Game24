import os

from datasets import DatasetDict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PaddingID = -100


# 数据预处理函数
def preprocess_function(examples):
    max_len = args.max_len
    prompt_template = '<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'
    system_prompt = ("Play 24 game. "
                     "Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations."
                     " Output the reasoning steps, one step per line. On the last line, output the expression, for example, "
                     "input: '10 1 12 3', "
                     "the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.")
    model_inputs = {'input_ids': [], 'labels': [], 'input_len': [], 'output_len': []}
    for i in range(len(examples['input'])):
        prompt = prompt_template.format(
            system_prompt=system_prompt,
            user_prompt=examples['input'][i]
        )
        a_ids = tokenizer.encode(prompt)
        b_ids = tokenizer.encode(f"{examples['output'][i]}", add_special_tokens=False) + [tokenizer.eos_token_id]
        context_length = len(a_ids)
        input_ids = a_ids + b_ids

        if max_len > len(input_ids):
            """
            使用 -100 填充
            """
            pad_length = max_len - len(input_ids)
            labels = [PaddingID] * context_length + b_ids + [PaddingID] * pad_length
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
        else:
            labels = [PaddingID] * context_length + b_ids
            labels = labels[:max_len]
            input_ids = input_ids[:max_len]

        model_inputs['input_ids'].append(input_ids)
        model_inputs['labels'].append(labels)
        model_inputs['input_len'].append(len(a_ids))
        model_inputs['output_len'].append(len(b_ids))
    return model_inputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./dataset/train/train.json")
    parser.add_argument("--output_dir", type=str, default="short")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-0.5B')
    args = parser.parse_args()
    
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("json", data_files={"train": args.dataset})

    # 应用数据预处理
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.01, seed=42)
    tokenized_datasets = DatasetDict({
        "train": tokenized_datasets["train"],
        "validation": tokenized_datasets["test"]
    })

    model = AutoModelForCausalLM.from_pretrained(model_name)
    training_args = TrainingArguments(
        output_dir="./checkpoint/" + args.output_dir,  # 检查点保存路径
        max_steps=args.max_steps,  # 最大训练步数
        eval_steps=args.eval_steps,  # 每 100 步进行评估
        evaluation_strategy="steps",  # 评估策略：按步数评估
        save_strategy="steps",  # 保存策略：按步数保存
        save_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        save_safetensors=True,
        save_only_model=True,
        metric_for_best_model="loss",  # 用验证集 loss 作为最优模型的指标
        greater_is_better=False,
        weight_decay=0.01,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        optim='adamw_hf',
        lr_scheduler_type="cosine",  # 学习率调度器：余弦退火
        per_device_train_batch_size=args.batch_size,  # 每个设备上的训练批次大小
        per_device_eval_batch_size=6,  # 每个设备上的评估批次大小
        gradient_accumulation_steps=args.accumulation_steps,  # 梯度累积步数（可根据显存调整）
        num_train_epochs=1,  # 暂不限制训练轮数，因为按 `max_steps` 控制
        fp16=True,  # 使用混合精度训练（如果硬件支持）
        logging_dir="./logs",  # 日志保存路径
        logging_steps=10,  # 每 10 步记录日志
        report_to=["tensorboard"],  # 日志报告工具（如 TensorBoard）
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # 开始训练
    trainer.train()
