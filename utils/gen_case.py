from itertools import combinations_with_replacement
from itertools import permutations

from tqdm import tqdm


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


def generate_24_game_combinations():
    # 生成1-13的所有四个数字的组合（允许重复）
    numbers = list(range(1, 14))
    combinations = list(combinations_with_replacement(numbers, 4))

    valid_sets = []

    tqdm_bar = tqdm(total=len(combinations))

    for candidate in combinations:
        if can_make_24(candidate):
            valid_sets.append(list(candidate))
        tqdm_bar.update(1)

    return valid_sets


def main():
    # 生成分组后的组合
    grouped_combinations = generate_24_game_combinations()
    print(len(grouped_combinations))


if __name__ == "__main__":
    main()
