from itertools import permutations


def solve_24(nums):
    """
    尝试用给定四个数字通过加、减、乘、除得到 24，
    如果可以，返回一种可行的计算表达式，否则返回 None。
    """
    # 容忍的浮点计算误差
    EPSILON = 1e-6

    # 用于在递归时存储 (值, 表达式)
    # 例如，(3.0, "3") 或 (7.5, "(3 + 4.5)")

    def dfs(cards):
        """
        参数 cards: [(值1, 表达式1), (值2, 表达式2), ...]
        如果能得到 24，返回对应表达式，否则返回 None
        """
        # 只剩一个数时，判断是否接近 24
        if len(cards) == 1:
            val, expr = cards[0]
            if abs(val - 24) < EPSILON:
                return expr
            else:
                return None

        # 尝试从当前卡片中任意选两张牌
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                a_val, a_expr = cards[i]
                b_val, b_expr = cards[j]

                # 剩下其它的卡
                remaining = [cards[k] for k in range(len(cards)) if k != i and k != j]

                # 可能的新结果集合 (值, 表达式)
                candidates = []

                # 加法
                candidates.append((a_val + b_val, f"({a_expr} + {b_expr})"))

                # 减法（两种顺序）
                candidates.append((a_val - b_val, f"({a_expr} - {b_expr})"))
                candidates.append((b_val - a_val, f"({b_expr} - {a_expr})"))

                # 乘法
                candidates.append((a_val * b_val, f"({a_expr} * {b_expr})"))

                # 除法（两种顺序，需保证除数不为 0）
                if abs(b_val) > EPSILON:
                    candidates.append((a_val / b_val, f"({a_expr} / {b_expr})"))
                if abs(a_val) > EPSILON:
                    candidates.append((b_val / a_val, f"({b_expr} / {a_expr})"))

                # 对于每一个候选结果，继续往下递归
                for val, expr in candidates:
                    new_cards = remaining + [(val, expr)]
                    res = dfs(new_cards)
                    if res is not None:
                        return res
        return None

    # 先对 nums 做所有排列，确保不同顺序都能被尝试
    for perm in set(permutations(nums)):
        # 每个数字都先包装成 (值, 表达式) 的形式
        initial_cards = [(float(x), str(x)) for x in perm]
        result_expr = dfs(initial_cards)
        if result_expr is not None:
            return result_expr

    # 所有排列和运算都尝试失败，返回 None
    return None


def compress_logs(log_lines, max_lines=10, keep_last_solution=True):
    """
    将原始的 24 点搜索日志（含各种运算和 roll back）进行压缩，返回一份更短的日志。
    参数：
      log_lines: List[str], 每个元素是一行日志文本
      max_lines: int, 压缩后想要保留的最大行数
      keep_last_solution: bool, 如果日志里出现多次 'reach 24! expression:'，是否保留最后一次出现？
                         若为 False，则保留第一次出现。
    返回：
      List[str]，压缩后的日志

    策略（简要）：
      1) 找到“reach 24! expression:” 那一行(可选是第一次或最后一次)，记为 pos。
         如果没找到这串文本，说明无解，则就保留全部(或自行决定怎么处理)。
      2) 截取 log_lines[0 .. pos] 这段作为 candidate。
      3) 如果 candidate 长度超过 max_lines，就把前半部分和后半部分各保留几行，中间插入一句 "...(some lines omitted)..."。
      4) 返回处理后的列表。
    """
    if not log_lines:
        return []

    # 1) 查找目标行下标：第一次 or 最后一次出现 "reach 24! expression:"
    solution_indices = [i for i, line in enumerate(log_lines) if "reach 24! expression:" in line]

    # 如果找不到，说明日志里没有成功解，可根据需求决定：要么全盘保留（相当于无解），要么空。
    if not solution_indices:
        # 这里的处理方式：直接返回日志全部或进行截断
        candidate = log_lines[:]  # 全部
        # 你也可以只返回空列表: return []
    else:
        if keep_last_solution:
            pos = solution_indices[-1]  # 保留最后一次出现
        else:
            pos = solution_indices[0]   # 保留第一次出现

        # 2) 截取从开头到 reach 24 的那行（含 pos）
        candidate = log_lines[: pos + 1]

    # 3) 如果 candidate 行数仍超过 max_lines，进行简单截断：
    #    a. 保留前 half = max_lines // 2 行
    #    b. 保留后 half = max_lines - half - 1 行（因为要空出 1 行放置 "...omitted..."）
    #    c. 中间用 "...(some lines omitted)..." 占位
    if len(candidate) > max_lines:
        half = max_lines // 2
        front_part = candidate[:half]
        back_part = candidate[-(max_lines - half - 1):]
        mid_line = f"...(some lines omitted, total={len(candidate)})..."
        return front_part + [mid_line] + back_part
    else:
        return candidate

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        (4, 7, 8, 8),  # 可以，如: (7 - 8/8)*4 = 24
        (1, 3, 4, 6),  # 可以，如: 6 / (1 - 3/4) = 24
        (1, 2, 3, 4),  # 很常见的示例
        (1, 1, 2, 4),
        (1, 1, 1, 1),
    ]

    for tc in test_cases:
        expr = solve_24(tc)
        print(tc, "=>", expr if expr else "无法得到24")