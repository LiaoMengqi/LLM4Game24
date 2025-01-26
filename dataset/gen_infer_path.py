import random
from fractions import Fraction


def fraction_str(x: Fraction) -> str:
    """
    将 Fraction 转化为字符串：
      - 若分母=1，则返回整数形式(如 '2')
      - 否则返回形如 '3/4' 的分数形式
    """
    if x.denominator == 1:
        return str(x.numerator)
    else:
        return f"{x.numerator}/{x.denominator}"


def parenthesize(a_expr: str, b_expr: str, op: str) -> str:
    """
    在字符串层面加上括号，如:
      parenthesize("1", "(3 / 4)", "-") -> "(1 - (3 / 4))"
    """
    return f"({a_expr} {op} {b_expr})"


def show_card(val_expr):
    """
    根据 (Fraction值, 表达式字符串) 来生成输出。

    规则：
    1) 如果表达式字符串与 fraction_str(...) 完全相同，说明它只是个单数字面量，如 "4" 或 "10"；
       则只返回 "4"、"10" 等简写，不再写成 "4 = 4"。
    2) 否则输出形如 "(1 + 3) = 4"、"((1 + 3) + 4) = 10" 等。
    """
    val, expr = val_expr
    frac_str = fraction_str(val)
    # 判断是否“表达式字符串本身”与 Fraction 化简后一致
    if expr == frac_str:
        # 这说明 expr 就是个单纯数字如 "4"
        return expr
    else:
        # 复合形式，显示 "(...)=..."
        return f"{expr} = {frac_str}"


def leftovers_to_str(list_of_cards):
    """
    将余下的 (val, expr) 列表转换为形如:
      "(1 + 3) = 4, 4, 6"
    的字符串。
    """
    return ", ".join(show_card(c) for c in list_of_cards)


def dfs_24(cards, steps) -> bool:
    """
    cards: [(Fraction数值, 表达式字符串), ...]
    steps: 搜索过程的日志列表

    如果能组成24，则返回True，并在steps最后追加 "reach 24! expression: XXX"
    否则返回False。
    """
    # 如果只剩一个数
    if len(cards) == 1:
        val, expr = cards[0]
        if val == Fraction(24, 1):
            steps.append(f"reach 24! expression: {expr}")
            return True
        else:
            return False

    n = len(cards)

    indices = []
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            indices.append((i, j))
    random.shuffle(indices)

    # 遍历所有 (i, j) 对，以及 +、-、*、/（两种减、两种除）
    for (i, j) in indices:
        a_val, a_expr = cards[i]
        b_val, b_expr = cards[j]

        a_val_str = '(' + fraction_str(a_val) + ')'
        b_val_str = '(' + fraction_str(b_val) + ')'

        remaining = [cards[k] for k in range(n) if k not in (i, j)]

        # 准备所有运算
        candidates = []

        # a + b
        s_val = a_val + b_val
        s_expr = parenthesize(a_expr, b_expr, "+")
        candidates.append((s_val, s_expr, f"{a_val_str} + {b_val_str}"))

        # a - b
        s_val = a_val - b_val
        s_expr = parenthesize(a_expr, b_expr, "-")
        candidates.append((s_val, s_expr, f"{a_val_str} - {b_val_str}"))

        # b - a
        s_val = b_val - a_val
        s_expr = parenthesize(b_expr, a_expr, "-")
        candidates.append((s_val, s_expr, f"{b_val_str} - {a_val_str}"))

        # a * b
        s_val = a_val * b_val
        s_expr = parenthesize(a_expr, b_expr, "*")
        candidates.append((s_val, s_expr, f"{a_val_str} * {b_val_str}"))

        # a / b
        if b_val != 0:
            s_val = a_val / b_val
            s_expr = parenthesize(a_expr, b_expr, "/")
            candidates.append((s_val, s_expr, f"{a_val_str} / {b_val_str}"))

        # b / a
        if a_val != 0:
            s_val = b_val / a_val
            s_expr = parenthesize(b_expr, a_expr, "/")
            candidates.append((s_val, s_expr, f"{b_val_str} / {a_val_str}"))

        # 打乱顺序保证随机性
        random.shuffle(candidates)
        for (val_res, expr_res, op_str) in candidates:
            new_cards = remaining + [(val_res, expr_res)]
            val_str = fraction_str(val_res)

            # 生成 “left: ...” 部分
            #   刚算出的那张牌在最前，其余在后
            expr_list = [(val_res, expr_res)] + remaining
            leftover_str = leftovers_to_str(expr_list)

            # 日志行，如 "1 + 3 = 4, left: (1 + 3) = 4, 4, 6"
            steps.append(
                f"{op_str} = {val_str}, left: {leftover_str}"
            )

            # 递归
            if dfs_24(new_cards, steps):
                return True
            else:
                # 回溯
                rollback_str = leftovers_to_str(cards)
                steps.append(f"roll back, left: {rollback_str}")

    return False


def solve_24_with_steps(nums):
    """
    输入4个数字(如 [1, 3, 4, 6]),
    返回包含“搜索和回溯日志”的列表 steps。

    若能最终组成24，会在 steps 末尾出现 "reach 24! expression: XXX"。
    否则返回 None。
    """
    # 确保是 (Fraction, str) 格式
    cards = [(Fraction(x), str(x)) for x in nums]
    steps = []

    # 先输出最初输入
    steps.append(" ".join(str(x) for x in nums))

    if dfs_24(cards, steps):
        return steps
    else:
        return None


if __name__ == "__main__":
    # 例子
    example = [1, 3, 4, 6]
    result = solve_24_with_steps(example)
    for step in result:
        print(step)
