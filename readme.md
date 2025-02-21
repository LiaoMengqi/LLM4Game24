[中文 README](readme_zh.md)

*Latest Updates*
- **[2025/02/02]** Uploaded the DeepSeek R1 Zero reinforcement learning pipeline code and experimental report, including implementations of GRPO and the rule-based reward model.  
- **[2025/01/29]** Optimized the efficiency of case generation and fine-tuning data generation algorithms. Training data inputs are now randomly shuffled to enhance data diversity. Updated the system prompt for subsequent reinforcement learning training. Added mixed-data fine-tuning and updated the new fine-tuning experimental report.  
- **[2025/01/26]** Uploaded SFT code and experimental report.  

# Introduction

This repository is a toy project for large-model Long Chain-of-Thought (Long CoT) fine-tuning and reinforcement learning in the context of solving the *24-point game*. It aims to provide an introductory practice platform for Long CoT fine-tuning and reinforcement learning with large models.

**About the 24-Point Game**:  
The goal of the 24-point game is to use four given numbers (ranging from 1 to 13) and basic arithmetic operations (addition, subtraction, multiplication, and division) to calculate a result equal to 24. For example, given the numbers `4, 2, 6, 3`, one possible solution is `(4 × 3) + (2 × 6) = 24`.

**Project Content**:  
- Provides code for generating Long CoT reasoning data, as well as fine-tuning and inference code for the Qwen model.  
- Implements the DeepSeek R1-Zero reinforcement learning pipeline to further enhance the model's performance in the 24-point game.  
- Details on data format design and experimental results can be found in the fine-tuning experimental report [`report/sft_report.pdf`](report/sft_report.pdf).  
- Experimental results for the reinforcement learning component are available in the reinforcement learning experimental report [`report/rl_report.pdf`](report/rl_report.pdf).  

This project is intended for educational and reference purposes only. Stay tuned for future updates!

**Future Plans**
- Implement DPO (Direct Preference Optimization) for preference learning.

## Fine-Tuning Data Generation

The data has already been generated and is stored in the `dataset/train` directory. The `train.json` file contains all training data, while `short.json`, `medium.json`, and `long.json` contain data for short, medium, and long reasoning paths, respectively. If you need to regenerate the data, you can run the following command:

```shell
python gen_sft_data.py
```

For generating experimental data in different formats:

```shell
python gen_abation_data.py
```

## Fine-Tuning and Evaluation

Fine-Tuning Command
```shell
CUDA_VISIBLE_DEVICES=0 python stf.py \
--dataset ./dataset/train/short.json \
--output_dir short_v3 \
--max_steps 1000 \
--batch_size 8 \
--accumulation_steps 4 \
--max_len 1250
```

Evaluation Command. Generating reasoning steps and results, with outputs saved in the `./result` directory:
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--path /root/autodl-tmp/ \
--checkpoint_id short_v3/checkpoint-900 \
--save_file short_v3.json \
--batch_size 16 \
--max_length 4096
```

The `result_statistic.ipynb` script provides code for calculating and visualizing metrics based on the generated results.

## Reinforcement Learning

For GRPO reinforcement learning, two devices are used to host the actor model and the reference model. If only one GPU is available, you can set `CUDA_VISIBLE_DEVICES=0` and specify `--ref_device 0`.

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

Parameter Descriptions:
- `--beta`: KL penalty coefficient.  
- `--max_momentum`: Used for dynamic reward normalization. The default is 0, indicating no dynamic normalization. Dynamic normalization is not used in DeepSeek Math.  
- `--batch_size`: Number of instances used in each iteration.  
- `--group_size`: Group size for GRPO, which corresponds to the number of samples generated per instance.  
- `--mini_batch_size`: Batch size for policy updates, affecting GPU memory usage.  
- `--lambda_oc`: Weight of the output correctness reward in the reward model.  
- `--lambda_pr`: Weight of the process correctness reward in the reward model.  
- `--lambda_l`: Weight of the length reward in the reward model.  
- `--top_p`: Top-p value for sampling candidate answers, default is 0.9.  
- `--temperature`: Temperature value for sampling candidate answers, default is 0.7.

## Acknowledgments

This project makes use of the following open-source libraries:

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

We are grateful to the authors and maintainers of this project for their amazing contributions to the open-source community.