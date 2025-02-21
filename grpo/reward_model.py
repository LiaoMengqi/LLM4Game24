from typing import List

import math
import torch
import torch.nn as nn


def extract_numbers(expr) -> list:
    curr_num = ''
    numbers = []
    for c in expr:
        if c.isdigit():
            curr_num += c
        elif curr_num:
            numbers.append(int(curr_num))
            curr_num = ''
    if curr_num:
        numbers.append(int(curr_num))
    return numbers


class RewardModel4Game24(nn.Module):
    def __init__(self,
                 lambda_oc: float = 0.08,
                 lambda_pr: float = 0.8,
                 lambda_l: float = 0.12,
                 length_threshold: int = 4,  # 最优路径
                 middle_offset: int = 20,
                 steepness: float = 0.2):
        super().__init__()
        self.lambda_pr = lambda_pr  # 步骤正确率
        self.lambda_l = lambda_l  # 长度奖励系数
        self.lambda_oc = lambda_oc  # 结果奖励系数
        self.length_threshold = length_threshold
        self.steepness = steepness
        self.middle_offset = middle_offset

    def forward(self,
                output: List[str],
                cases
                ):
        res = [self.evaluate_result(output[i], cases[i]) for i in range(len(output))]
        res = torch.Tensor(res).transpose(0, 1)
        rewards = res[0]
        correct = res[1]
        # print(rewards.shape)
        return rewards, correct

    def length_reward_sigmoid(self, length):
        """S型递减的长度奖励"""
        if length <= self.length_threshold:
            return 1.0
        return 1 / (1 + math.exp(self.steepness * (length - self.length_threshold - self.middle_offset)))

    def check_solution(self, lines: List[str], print_error=False):

        """检查完整解决方案"""

        # lines = text.strip().split('\n')
        start = lines[0]
        lines = lines[1:]
        error_lines = 0
        outcome_reward = 0
        # correct_reward = 0
        for i, line in enumerate(lines):

            flag = True
            if 'left:' in line:
                if 'roll' not in line and line in lines[:i]:  # 计算动作重复惩罚
                    error_lines += 1
                    continue
                exprs = line.split('left:')[1].strip()
                if ',' in exprs or '=' in exprs:
                    exprs = exprs.split(',')
                else:
                    exprs = exprs.split(' ')
                numbers = []
                for expr in exprs:
                    if '=' in expr:
                        left_expr = expr.split('=')[0]
                        try:
                            right_value = eval(expr.split('=')[1])
                        except Exception as e:
                            error_lines += 1
                            flag = False
                            break
                        expr_numbers = extract_numbers(left_expr)
                        numbers += expr_numbers
                        try:
                            value = eval(left_expr)
                            if abs(value - right_value) > 1e-6:
                                error_lines += 1
                                flag = False
                                break
                        except Exception as e:
                            error_lines += 1
                            flag = False
                            break
                    else:
                        try:
                            value = int(expr)
                            numbers.append(value)
                        except Exception as e:
                            error_lines += 1
                            flag = False
                            break
            elif 'expression:' in line:
                if not i == len(lines) - 1:  # 冗余后缀惩罚
                    error_lines += 1
                    continue

                expression = line.split('expression:')[-1]
                numbers = extract_numbers(expression)
                try:
                    value = eval(expression)
                    if abs(value - 24) > 1e-6:  # 数值错误
                        error_lines += 1
                        flag = False
                except Exception as e:  # 表达式错误
                    flag = False
                    error_lines += 1
                case = sorted([int(i) for i in start.split(' ')])
                numbers = sorted(numbers)
                if flag and numbers == case:
                    outcome_reward = 1
                else: # 数字不匹配
                    error_lines += 1
                    outcome_reward = 0
                continue
            else:
                error_lines += 1
                continue
            case = sorted([int(i) for i in start.split(' ')])
            numbers = sorted(numbers)
            if flag and not numbers == case:
                error_lines += 1
                flag = False
            if not flag and print_error:
                print(f"Error in line : {line}")
        return 1 - error_lines / len(lines), outcome_reward

    def evaluate_result(self, result, case):

        output = result.split('assistant\n')[-1]
        start = ' '.join([str(i) for i in case]) + '\n'
        lines = (start + output).split('\n')
        reward_pr, reward_oc = self.check_solution(lines)
        reward_l = self.length_reward_sigmoid(len(lines) - 1)
        reward = self.lambda_pr * reward_pr + self.lambda_l * reward_l + self.lambda_oc * reward_oc

        return reward, 1 if reward_oc == 1 else 0


if __name__ == '__main__':
    model = RewardModel4Game24()
    result = [
        "system\nPlay 24 game. Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations. Output the reasoning steps, one step per line. On the last line, output the expression, for example, input: '10 1 12 3', the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.\nuser\n5 7 9 13\nassistant\n(9) / (13) = 9/13, left: (9 / 13) = 9/13, 5, 7\n(9/13) / (5) = 9/65, left: ((9 / 13) / 5) = 9/65, 7\n(7) + (9/65) = 472/65, left: (7 + ((9 / 13) / 5)) = 472/65",
        "system\nPlay 24 game. Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations. Output the reasoning steps, one step per line. On the last line, output the expression, for example, input: '10 1 12 3', the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.\nuser\n13 2 2 2\n\nassistant\n(2) / (2) = 1, left: (2 / 2) = 1, 13, 2\n(13) - (1) = 12, left: (13 - (2 / 2)) = 12, 2\n(2) * (12) = 24, left: (2 * (13 - (2 / 2))) = 24\nreach 24! expression: (2 * (13 - (2 / 2)))",
        "system\nPlay 24 game. Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations. Output the reasoning steps, one step per line. On the last line, output the expression, for example, input: '10 1 12 3', the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.\nuser\n8 10 8 7\n\nassistant\n(8) - (7) = 1, left: (8 - 7) = 1, 8, 10\n(10) - (8) = 2, left: (10 - 8) = 2, (8 - 7) = 1\nroll back, left: (8 - 7) = 1, 8, 10\nroll back, left: 8 10 8 7\n(8) / (7) = 8/7, left: (8 / 7) = 8/7, 8, 10\n(8) + (10) = 18, left: (8 + 10) = 18, (8 / 7) = 8/7\n(8/7) * (18) = 24, left: ((8 / 7) * (8 + 10)) = 24\nreach 24! expression: ((8 / 7) * (8 + 10))",
        "system\nPlay 24 game. Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations. Output the reasoning steps, one step per line. On the last line, output the expression, for example, input: '10 1 12 3', the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.\nuser\n1 6 10 5\n\nassistant\n(1) + (5) = 6, left: (1 + 5) = 6, 6, 10\n(10) - (6) = 4, left: (10 - 6) = 4, (1 + 5) = 6\n(6) * (4) = 24, left: ((1 + 5) * (10 - 6)) = 24\nreach 24! expression: ((1 + 5) * (10 - 6)",
        "system\nPlay 24 game. Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations. Output the reasoning steps, one step per line. On the last line, output the expression, for example, input: '10 1 12 3', the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.\nuser\n4 1 9 4\n\nassistant\n(4) - (9) = -5, left: (4 - 9) = -5, 4, 1\n(1) - (-5) = 6, left: (1 - (4 - 9)) = 6, 4\n(4) * (6) = 24, left: (4 * (1 - (4 - 9))) = 24\nreach 24! expression: (4 * (1 - (9 - 9)))",
        "system\nPlay 24 game. Given four numbers, determine if it's possible to reach 24 through basic arithmetic operations. Output the reasoning steps, one step per line. On the last line, output the expression, for example, input: '10 1 12 3', the last line should output: 'reach 24! expression: ((12 + 3) + (10 - 1))'.\nuser\n9 6 6 6\n\nassistant\n(9) + (6) = 15, left: (9 + 6) = 15, 6, 6\n(6) / (15) = 2/5, left: (6 / (9 + 6)) = 2/5, 6\nroll back, left: (9 + 6) = 15, 6, 6\n(6) + (15) = 21, left: (6 + (9 + 6)) = 21, 6\nroll back, left: (9 + 6) = 15, 6, 6\n(6) - (15) = -9, left: (6 - (9 + 6)) = -9, 6\nroll back, left: (9 + 6) = 15, 6, 6\n(6) * (15) = 90, left: (6 * (9 + 6)) = 90, 6\nroll back, left: (9 + 6) = 15, 6, 6\n(6) / (6) = 1, left: (6 / 6) = 1, (9 + 6) = 15\nroll back, left: (9 + 6) = 15, 6, 6\n(6) - (6) = 0, left: (6 - 6) = 0, (9 + 6) = 15\nroll back, left: (9 + 6) = 15, 6, 6\n(6) + (6) = 12, left: (6 + 6) = 12, (9 + 6) = 15\nroll back, left: (9 + 6) = 15, 6, 6\n(6) / (6) = 1, left: (6 / 6) = 1, (9 + 6) = 15\nroll back, left: (9 + 6) = 15, 6, 6\n(6) * (6) = 36, left: (6 * 6) = 36, (9 + 6) = 15\nroll back, left: (9 + 6) = 15, 6, 6\nroll back, left: 9 6 6 6\n(9) - (6) = 3, left: (9 - 6) = 3, 6, 6\n(6) * (3) = 18, left: (6 * (9 - 6)) = 18, 6\n(6) + (18) = 24, left: (6 + (6 * (9 - 6))) = 24",

    ]
    cases = [[5, 7, 9, 13],
             [13, 2, 2, 2],
             [8, 10, 8, 7],
             [1, 6, 10, 5],
             [4, 1, 9, 4],
             [9, 6, 6, 6],
             ]
    rm = RewardModel4Game24()
    rw, cr = rm.forward(result, cases)
    print((rw - torch.mean(rw)) / torch.std(rw))
    print(cr.sum() / 6)
