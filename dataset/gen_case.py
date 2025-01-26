import json
import random
from itertools import permutations


def can_make_24(nums):
    """
    给定四个数字，判断是否能通过加、减、乘、除（允许任意顺序和括号组合）得到 24。
    采用回溯的思路。
    """
    # 误差阈值
    EPSILON = 1e-6

    # 长度检查
    if len(nums) != 4:
        return False

    def dfs(cards):
        # 如果只剩一个数字，判断它是否与 24 足够接近
        if len(cards) == 1:
            return abs(cards[0] - 24) < EPSILON

        # 从当前卡片中选两张牌
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                a = cards[i]
                b = cards[j]
                # 剩余卡片
                remaining = [cards[k] for k in range(len(cards)) if k != i and k != j]

                # a,b 可以生成的所有可能新数
                new_nums = []
                new_nums.append(a + b)
                new_nums.append(a - b)
                new_nums.append(b - a)
                new_nums.append(a * b)
                if abs(b) > EPSILON:
                    new_nums.append(a / b)
                if abs(a) > EPSILON:
                    new_nums.append(b / a)

                # 对每个新数，继续递归
                for num in new_nums:
                    if dfs(remaining + [num]):
                        return True
        return False

    # 尝试所有排列，只要有一个排列能实现 24 就返回 True
    for perm in set(permutations(nums)):
        if dfs(list(perm)):
            return True
    return False


def generate_valid_sets_24(n, lower=1, upper=13):
    """
    随机生成 n 组“24点有效案例”。
    参数说明：
      n: 需要的案例数量
      lower, upper: 生成数字的区间（默认1~13）
    返回: 一个列表，每个元素是可行的一组数字 [a, b, c, d]
    """
    valid_sets = []
    seen = set()

    while len(valid_sets) < n:
        # 在 [lower, upper] 区间生成 4 个随机数
        candidate = [random.randint(lower, upper) for _ in range(4)]

        # 判断是否能组成 24
        if can_make_24(candidate):
            # 为了去重，按排序后的 tuple 来判断
            key = tuple(sorted(candidate))
            if key not in seen:
                seen.add(key)
                valid_sets.append(candidate)

    return valid_sets


if __name__ == "__main__":
    # 测试：生成 10 组有效案例
    examples = generate_valid_sets_24(n=1100, upper=13)
    with open("24_case.json", "w") as f:
        json.dump(examples, f)
    # 打印结果
    # print("生成的有效 24 点案例：")
    # for e in examples:
    #     print(e)
