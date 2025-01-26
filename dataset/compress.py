import random
from typing import List, Optional


class LogNode:
    """
    双向 DFS 树节点示例，每个节点存储:
      • line: 日志行内容
      • parent: 父节点(可选)
      • children: 子节点列表
      • subtree_size: 以本节点为根的整颗子树节点总数
      • is_solution, keep, closed: 可根据应用需求设置的标记属性
    """

    def __init__(self, line: str):
        self.line: str = line
        self.parent: Optional['LogNode'] = None
        self.children: List['LogNode'] = []

        self.is_solution: bool = False
        self.keep: bool = False
        self.closed: bool = False

        # 注意：只做占位初始化，真正的准确值要在 build_tree 完成后，
        # 用 compute_subtree_sizes(...) 进行后序遍历来计算。
        self.subtree_size: int = 1

    def __repr__(self):
        return f"<LogNode line={self.line!r}, sz={self.subtree_size}, keep={self.keep}, sol={self.is_solution}>"


def build_tree_from_logs(log_lines: List[str]) -> LogNode:
    """
    简易示例：根据日志行模拟 DFS 构建树。
      - 遇到 "reach 24! expression:" 说明这是个解节点 is_solution=True
      - 遇到 "roll back" 就 pop 栈顶，并记录个 'closed' 节点
      - 其他行就是普通搜索过程
    """
    root = LogNode("<root>")
    stack = [root]

    for line in log_lines:
        if "reach 24! expression:" in line:
            node = LogNode(line)
            node.is_solution = True
            node.parent = stack[-1]
            stack[-1].children.append(node)
            stack.append(node)

        elif "roll back" in line:
            if len(stack) > 1:
                popped = stack.pop()
                popped.closed = True

        else:
            node = LogNode(line)
            node.parent = stack[-1]
            stack[-1].children.append(node)
            stack.append(node)

    return root


def mark_solution_path(root: LogNode) -> None:
    """
    对整棵树做 DFS，凡是遇到 is_solution=True 的节点，
    将从根到该节点的整条路径都标记 keep=True。
    """

    def dfs(node: LogNode, path_stack: List[LogNode]):
        path_stack.append(node)
        if node.is_solution:
            # 整条路径所有节点都标记为 keep=True
            for p in path_stack:
                p.keep = True
        size = 1
        for ch in node.children:
            size += dfs(ch, path_stack)
        node.subtree_size = size
        path_stack.pop()
        return size

    dfs(root, [])


def gather_deletable_nodes(root: LogNode) -> List[LogNode]:
    """
    收集所有可删除的节点，这里限定为 keep=False 的节点(不在解路径上)。
    因此解路径(keep=True)上的任何节点都不会被删除。
    """
    stack = [root]
    results = []
    while stack:
        node = stack.pop()
        for ch in node.children:
            stack.append(ch)

        # 忽略 <root>
        if node is not root:
            # 只有 keep=False 的节点才会被考虑删除
            if not node.keep:
                results.append(node)
    return results


def compute_subtree_sizes(node: LogNode) -> int:
    """
    后序遍历：计算该 node 的 subtree_size = 1(自己) + sum(各子节点的 subtree_size)。
    返回 node 的 subtree_size。
    """
    if not node.children:
        node.subtree_size = 1
        return 1

    total = 1
    for ch in node.children:
        total += compute_subtree_sizes(ch)
    node.subtree_size = total
    return total


def remove_subtree(subtree_root: LogNode) -> int:
    """
    将 subtree_root 这棵子树从其父节点中删除，返回实际删除的节点数(= subtree_root.subtree_size)。
    """
    removed_count = subtree_root.subtree_size
    parent = subtree_root.parent
    if parent is not None:
        parent.children.remove(subtree_root)
        # subtree_root.parent = None
    return removed_count


def update_subtree_size_upwards(start_node: LogNode, removed_count: int) -> None:
    """
    当删除 start_node (或其子树) 时，需要向上通知所有祖先节点的 subtree_size -= removed_count。
    直到没有父节点(到达根)为止。
    """
    cur = start_node.parent
    while cur is not None:
        cur.subtree_size -= removed_count
        cur = cur.parent


def flatten_tree_to_logs(node: LogNode) -> List[str]:
    """
    前序遍历整棵树，收集所有节点的 line 文本(忽略 <root>)。
    """
    output = []

    def preorder(nd: LogNode):
        if nd.line != "<root>":
            output.append(nd.line)
        for c in nd.children:
            preorder(c)
        if not nd.keep and nd.parent:
            output.append("roll back, left: " + nd.parent.line.split(": ")[-1])

    preorder(node)
    return output


def iterative_delete_smallest_subtrees(root: LogNode, max_lines: int):
    """
    逻辑：
      1) 先对整棵树 compute_subtree_sizes，得到准确初始化。
      2) 判断 (root.subtree_size - 1) 是否 <= max_lines (因为 <root>不输出)
         - 若满足则不需要删了
         - 若不满足则收集可删节点(示例: 非解节点)
           => 按 subtree_size 从小到大删除一个
           => update_subtree_size_upwards
           => 继续循环，直到达到 max_lines 或无可删节点
    """
    # (1) 先初始化 subtree_size
    # compute_subtree_sizes(root)
    candidates = gather_deletable_nodes(root)
    while True:
        current_count = root.subtree_size - 1  # 不包含根节点
        if current_count <= max_lines:
            break
        if not candidates:
            break

        # 找到 subtree_size 最小的那一个
        random.shuffle(candidates)  # 打乱顺序保证随机性
        candidates.sort(key=lambda n: n.subtree_size)
        smallest = candidates[0]

        removed_num = remove_subtree(smallest)
        candidates = candidates[1:]
        # 虽然 smallest 已经被 detach，但可以通过 smallest.parent 往上更新
        update_subtree_size_upwards(smallest, removed_num)
        # 重复检查，直至满足或者 candidates 为空


def compress_search_logs(logs, max_lines):
    root = build_tree_from_logs(logs)
    # compute_subtree_sizes(root)
    mark_solution_path(root)
    iterative_delete_smallest_subtrees(root, max_lines)
    compressed = flatten_tree_to_logs(root)
    return compressed


def demo():
    logs = [
        "5 9 13 7",
        "(5) * (9) = 45, left: (5 * 9) = 45, 13, 7",
        "(13) / (7) = 13/7, left: (13 / 7) = 13/7, (5 * 9) = 45",
        "(45) + (13/7) = 328/7, left: ((5 * 9) + (13 / 7)) = 328/7",
        "roll back, left: (13 / 7) = 13/7, (5 * 9) = 45",
        "(45) * (13/7) = 585/7, left: ((5 * 9) * (13 / 7)) = 585/7",
        "roll back, left: (13 / 7) = 13/7, (5 * 9) = 45",
        "(13/7) - (45) = -302/7, left: ((13 / 7) - (5 * 9)) = -302/7",
        "roll back, left: (13 / 7) = 13/7, (5 * 9) = 45",
        "(45) / (13/7) = 315/13, left: ((5 * 9) / (13 / 7)) = 315/13",
        "roll back, left: (13 / 7) = 13/7, (5 * 9) = 45",
        "(13/7) / (45) = 13/315, left: ((13 / 7) / (5 * 9)) = 13/315",
        "roll back, left: (13 / 7) = 13/7, (5 * 9) = 45",
        "(45) - (13/7) = 302/7, left: ((5 * 9) - (13 / 7)) = 302/7",
        "roll back, left: (13 / 7) = 13/7, (5 * 9) = 45",
        "roll back, left: (5 * 9) = 45, 13, 7",
        "(13) * (7) = 91, left: (13 * 7) = 91, (5 * 9) = 45",
        "(45) + (91) = 136, left: ((5 * 9) + (13 * 7)) = 136",
        "roll back, left: (13 * 7) = 91, (5 * 9) = 45",
        "(45) / (91) = 45/91, left: ((5 * 9) / (13 * 7)) = 45/91",
        "roll back, left: (13 * 7) = 91, (5 * 9) = 45",
        "roll back, left: (5 * 9) = 45, 13, 7",
        "roll back, left: 5 13 7 9",
        "(9) - (5) = 4, left: (9 - 5) = 4, 13, 7",
        "(7) + (4) = 11, left: (7 + (9 - 5)) = 11, 13",
        "(13) + (11) = 24, left: (13 + (7 + (9 - 5))) = 24",
        "reach 24! expression: (13 + (7 + (9 - 5)))"
    ]

    # 删除后输出结果
    compressed = compress_search_logs(logs, 8)
    # print(f"\nAfter delete, want <= {max_lines} lines, actual = {len(compressed)} lines:\n")
    for line in compressed:
        print("   ", line)


if __name__ == "__main__":
    demo()
