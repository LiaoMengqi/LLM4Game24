from typing import List


class RewardModel4Game24:
    def __init__(self,
                 lambda_cs: float = 1,
                 lambda_f: float = 1,
                 lambda_cr: float = 1,
                 lambda_l: float = 1,
                 length_threshold: int = 20,
                 device: str = "cpu"):
        self.device = device
        self.lambda_cs = lambda_cs  # 一致性奖励系数
        self.lambda_cr = lambda_cr  # 正确性奖励系数
        self.lambda_l = lambda_l  # 长度奖励系数
        self.lambda_f = lambda_f  # 完成或格式奖励系数
        self.length_threshold = length_threshold

    def forward(self, output: List[str], cases):

        pass

    def length_score(self, length: int) -> float:
        """
        返回长度得分
        """
        return self.lambda_l * len(length)

    def evaluate_result(self, result, case) -> float:
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

        lines = result.split('\n')
        last_line = lines[-1]
        if "expression:" not in last_line:
            # 没有完成或格式错误
            return 0
        else:
            finish_reward = 1
        length_reward = self.length_score(len(lines))
        expression = last_line.split('expression:')[-1]
        nums = extract_numbers(expression)
        if not case.sort() == nums.sort():
            # 数字不匹配
            consistent_reward = 0
            correct_reward = 0
        else:
            try:
                value = eval(expression)
                if abs(value - 24) < 1e-6:
                    correct_reward = 1
                else:
                    # 计算结果不对
                    correct_reward = 0
            except Exception as e:
                # 表达式解析错误
                correct_reward = -0.5
            consistent_reward = 1
        return self.lambda_cs * consistent_reward + \
            self.lambda_cr * correct_reward + \
            self.lambda_f * finish_reward + \
            self.lambda_l * length_reward
