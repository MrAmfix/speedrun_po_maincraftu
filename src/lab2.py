# My tasks: 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
# Points: 1 + 2.5 + 2 + 2 + 2 + 2 + 2 + 3 + 3 + 3 + 3 + 3 = 28.5
from typing import TextIO, List, Optional
from utils.decorators import fin_fout, performance
from utils.AVLTree import *
from collections import deque


@performance
@fin_fout
def lab2_task3(fin: TextIO, fout: TextIO):
    class Tree:
        def __init__(self, value=0):
            self.value = value
            self.left: Optional[Tree] = None
            self.right: Optional[Tree] = None

        def add(self, value):
            if self.value > value:
                if self.left is None:
                    self.left = Tree(value)
                else:
                    self.left.add(value)
            elif self.value < value:
                if self.right is None:
                    self.right = Tree(value)
                else:
                    self.right.add(value)

        def search(self, value) -> int:
            if value < self.value:
                if self.left is None:
                    return self.value
                else:
                    return self.left.search(value)
            else:
                if self.right is None:
                    return 0
                else:
                    return self.right.search(value)

    queries = fin.read().split('\n')
    tree = Tree()
    for query in queries:
        if query[0] == '+':
            tree.add(int(query[2:]))
        else:
            fout.write(f'{str(tree.search(int(query[2:])))}\n')


@performance
@fin_fout
def lab2_task7(fin: TextIO, fout: TextIO):
    class TreeNode:
        def __init__(self, value, left, right):
            self.value = value
            self.left = left
            self.right = right

    def is_bst(nodes: List[TreeNode]):
        stack = [(nodes[0], float('-inf'), float('inf'))]
        while stack:
            node, min_val, max_val = stack.pop()
            if not (min_val <= node.value < max_val):
                return False
            if node.left != -1:
                stack.append((nodes[node.left], min_val, node.value))
            if node.right != -1:
                stack.append((nodes[node.right], node.value, max_val))
        return True

    n = int(fin.readline())
    if n == 0:
        fout.write('CORRECT')
    else:
        nodes = [TreeNode(*list(map(int, fin.readline().split()))) for _ in range(n)]
        fout.write('CORRECT' if is_bst(nodes) else 'INCORRECT')


@performance
@fin_fout
def lab2_task8(fin: TextIO, fout: TextIO):
    class AVLNode:
        def __init__(self, value, left, right):
            self.value = value
            self.left = left
            self.right = right
            self.height = None

    def set_height(node):
        stackq = [node]
        while stackq:
            curr_node = stackq[-1]
            flag = True
            if curr_node.left is not None and curr_node.left.height is None:
                stackq.append(curr_node.left)
                flag = False
            if curr_node.right is not None and curr_node.right.height is None:
                stackq.append(curr_node.right)
                flag = False
            if flag:
                lh = 0 if curr_node.left is None else curr_node.left.height
                rh = 0 if curr_node.right is None else curr_node.right.height
                curr_node.height = max(lh, rh) + 1
                stackq.pop()

    n = int(fin.readline())
    vertices = [AVLNode(None, None, None) for _ in range(n)]
    for i in range(n):
        v, l, r = map(int, fin.readline().split())
        vertices[i].value = v
        if l != 0:
            vertices[i].left = vertices[l - 1]
        if r != 0:
            vertices[i].right = vertices[r - 1]
    set_height(vertices[0])
    fout.write(str(vertices[0].height))


@performance
@fin_fout
def lab2_task9(fin: TextIO, fout: TextIO):
    def full_delete(node, value):
        if node is None:
            return 0
        if value < node.value and node.left is not None:
            if node.left.value == value:
                d = len_delete_part(node.left)
                node.left = None
                return d
            else:
                return full_delete(node.left, value)
        elif value > node.value and node.right is not None:
            if node.right.value == value:
                d = len_delete_part(node.right)
                node.right = None
                return d
            else:
                return full_delete(node.right, value)
        else:
            return 0

    def len_delete_part(node):
        if node is None:
            return 0
        return len_delete_part(node.left) + 1 + len_delete_part(node.right)

    n = int(fin.readline())
    vertices = [Node(None) for _ in range(n)]
    for i in range(n):
        v, l, r = map(int, fin.readline().split())
        vertices[i].value = v
        if l != 0:
            vertices[i].left = vertices[l - 1]
        if r != 0:
            vertices[i].right = vertices[r - 1]
    root = vertices[0]
    m = int(fin.readline())
    delete_numbers = list(map(int, fin.readline().split()))
    for num in delete_numbers:
        n -= full_delete(root, num)
        fout.write(f'{n}\n')


@performance
@fin_fout
def lab2_task10(fin: TextIO, fout: TextIO):
    def is_binary_search_tree(node: Node):
        deq = deque()
        deq.append((node, None, None))  # node, value of parent, 0 - smaller, 1 - bigger
        while deq:
            curr_node, parent_val, typ = deq.popleft()
            if typ is None:
                deq.append((curr_node.left, curr_node.value, False))
                deq.append((curr_node.right, curr_node.value, True))
            elif curr_node is not None:
                if not typ:
                    if curr_node.value >= parent_val:
                        return False
                else:
                    if curr_node.value <= parent_val:
                        return False
                deq.append((curr_node.left, curr_node.value, False))
                deq.append((curr_node.right, curr_node.value, True))
        return True


    n = int(fin.readline())
    if n == 0:
        fout.write('YES')
        return
    vertices = [Node(None) for _ in range(n)]
    for i in range(n):
        v, l, r = map(int, fin.readline().split())
        vertices[i].value = v
        if l != 0:
            vertices[i].left = vertices[l - 1]
        if r != 0:
            vertices[i].right = vertices[r - 1]
    root = vertices[0]
    fout.write('YES' if is_binary_search_tree(root) else 'NO')


@performance
@fin_fout
def lab2_task11(fin: TextIO, fout: TextIO):
    queries = [i.strip().split() for i in fin.readlines()]
    tree = AVLTree()
    root = None
    for typ, num in queries:
        if typ == 'insert':
            root = tree.insert(root, int(num))
        elif typ == 'delete':
            root = tree.delete(root, int(num))
        elif typ == 'exists':
            if tree.find_elem(root, int(num)):
                fout.write('true\n')
            else:
                fout.write('false\n')
        elif typ == 'next':
            nxt = tree.next(root, int(num))
            if nxt is None:
                fout.write('none\n')
            else:
                fout.write(f'{nxt}\n')
        elif typ == 'prev':
            prv = tree.prev(root, int(num))
            if prv is None:
                fout.write('none\n')
            else:
                fout.write(f'{prv}\n')


@performance
@fin_fout
def lab2_task12(fin: TextIO, fout: TextIO):
    class AVLNode:
        def __init__(self, value, left, right):
            self.value = value
            self.left = left
            self.right = right
            self.height = None

    def get_balance(node) -> int:
        return (- (0 if node.left is None else node.left.height) +
                (0 if node.right is None else node.right.height))

    def set_height(node):
        stackq = [node]
        while stackq:
            curr_node = stackq[-1]
            flag = True
            if curr_node.left is not None and curr_node.left.height is None:
                stackq.append(curr_node.left)
                flag = False
            if curr_node.right is not None and curr_node.right.height is None:
                stackq.append(curr_node.right)
                flag = False
            if flag:
                lh = 0 if curr_node.left is None else curr_node.left.height
                rh = 0 if curr_node.right is None else curr_node.right.height
                curr_node.height = max(lh, rh) + 1
                stackq.pop()


    n = int(fin.readline())
    nodes = [AVLNode(None, None, None) for _ in range(n)]

    for i in range(n):
        value, l, r = list(map(int, fin.readline().split()))
        nodes[i].value = value
        nodes[i].left = None if l == 0 else nodes[l - 1]
        nodes[i].right = None if r == 0 else nodes[r - 1]

    set_height(nodes[0])
    for i in range(n):
        fout.write(f'{get_balance(nodes[i])}\n')


@performance
@fin_fout
def lab2_task13(fin: TextIO, fout: TextIO):
    def writer(node: Node):
        ind = 2
        deq = deque()
        deq.append(node)
        while deq:
            curr_node = deq.popleft()
            s = f'{curr_node.value} '
            if curr_node.left is not None:
                deq.append(curr_node.left)
                s += f'{ind} '
                ind += 1
            else:
                s += '0 '

            if curr_node.right is not None:
                deq.append(curr_node.right)
                s += f'{ind}\n'
                ind += 1
            else:
                s += '0\n'
            fout.write(s)

    n = int(fin.readline())
    tree = AVLTree()
    vertices = [Node(None) for _ in range(n)]
    for i in range(n):
        v, l, r = map(int, fin.readline().split())
        vertices[i].value = v
        if l != 0:
            vertices[i].left = vertices[l - 1]
        if r != 0:
            vertices[i].right = vertices[r - 1]
    root = vertices[0]
    if root.value > root.right.value:
        root = tree.left_rotate(root)
    else:
        root.right = tree.right_rotate(root.right)
        root = tree.left_rotate(root)
    fout.write(f'{n}\n')
    writer(root)


@performance
@fin_fout
def lab2_task14(fin: TextIO, fout: TextIO):
    def writer(node: Node):
        ind = 2
        deq = deque()
        deq.append(node)
        while deq:
            curr_node = deq.popleft()
            s = f'{curr_node.value} '
            if curr_node.left is not None:
                deq.append(curr_node.left)
                s += f'{ind} '
                ind += 1
            else:
                s += '0 '

            if curr_node.right is not None:
                deq.append(curr_node.right)
                s += f'{ind}\n'
                ind += 1
            else:
                s += '0\n'
            fout.write(s)

    n = int(fin.readline())
    tree = AVLTree()
    vertices = [Node(None) for _ in range(n)]
    for i in range(n):
        v, l, r = map(int, fin.readline().split())
        vertices[i].value = v
        if l != 0:
            vertices[i].left = vertices[l - 1]
        if r != 0:
            vertices[i].right = vertices[r - 1]
    root = vertices[0]
    fout.write(f'{n + 1}\n')
    writer(tree.insert(root, int(fin.readline())))


@performance
@fin_fout
def lab2_task15(fin: TextIO, fout: TextIO):
    def writer(node: Node):
        ind = 2
        deq = deque()
        deq.append(node)
        while deq:
            curr_node = deq.popleft()
            s = f'{curr_node.value} '
            if curr_node.left is not None:
                deq.append(curr_node.left)
                s += f'{ind} '
                ind += 1
            else:
                s += '0 '

            if curr_node.right is not None:
                deq.append(curr_node.right)
                s += f'{ind}\n'
                ind += 1
            else:
                s += '0\n'
            fout.write(s)

    n = int(fin.readline())
    tree = AVLTree()
    vertices = [Node(None) for _ in range(n)]
    for i in range(n):
        v, l, r = map(int, fin.readline().split())
        vertices[i].value = v
        if l != 0:
            vertices[i].left = vertices[l - 1]
        if r != 0:
            vertices[i].right = vertices[r - 1]
    root = vertices[0]
    fout.write(f'{n - 1}\n')
    writer(tree.delete(root, int(fin.readline())))


@performance
@fin_fout
def lab2_task16(fin: TextIO, fout: TextIO):
    n = int(fin.readline())
    queries = [fin.readline().split() for _ in range(n)]
    tree = AVLTree()
    root = None
    for opera, value in queries:
        if opera == '+1':
            root = tree.insert(root, int(value))
        elif opera == '-1':
            root = tree.delete(root, int(value))
        else:
            fout.write(f'{tree.k_max(root, int(value), set())}\n')


@performance
@fin_fout
def lab2_task17(fin: TextIO, fout: TextIO):
    class SplayNode:
        def __init__(self, value: int):
            self.value = value
            self.left = None
            self.right = None

    class SplayTree:
        def __init__(self):
            self.root = None

        def insert(self, value: int):
            if self.root is None:
                self.root = SplayNode(value)
                return

            self.root = self.splay_and_rebalance(value, self.root)

            if value < self.root.value:
                node = SplayNode(value)
                node.right = self.root
                node.left = self.root.left
                self.root.left = None
                self.root = node
            elif value > self.root.value:
                node = SplayNode(value)
                node.left = self.root
                node.right = self.root.right
                self.root.right = None
                self.root = node

        def delete(self, value: int):
            if self.find(value) is None or self.root is None:
                return

            if self.root.left is None:
                self.root = self.root.right
            else:
                temp = self.root.right
                self.root = self.root.left
                self.splay_and_rebalance(value, self.root)
                self.root.right = temp

        def find(self, value: int) -> bool:
            self.root = self.splay_and_rebalance(value, self.root)
            return self.root is not None and self.root.value == value

        def sum(self, l, r) -> int:
            def recursive_sum(node: SplayNode, l: int, r: int):
                if node is None:
                    return 0
                if node.value < l:
                    return recursive_sum(node.right, l, r)
                elif node.value > r:
                    return recursive_sum(node.left, l, r)
                else:
                    return node.value + recursive_sum(node.left, l, r) + recursive_sum(node.right, l, r)
            return recursive_sum(self.root, l, r)

        def rotate_left(self, node: SplayNode):
            right = node.right
            node.right = right.left
            right.left = node
            return right

        def rotate_right(self, node: SplayNode):
            left = node.left
            node.left = left.right
            left.right = node
            return left

        def splay_and_rebalance(self, value: int, node: SplayNode):
            if node is None or node.value == value:
                return node

            if value < node.value:
                if node.left is None:
                    return node
                if value < node.left.value:
                    node.left.left = self.splay_and_rebalance(value, node.left.left)
                    node = self.rotate_right(node)
                elif value > node.left.value:
                    node.left.right = self.splay_and_rebalance(value, node.left.right)
                    if node.left.right is not None:
                        node.left = self.rotate_left(node.left)
                return self.rotate_right(node) if node.left is not None else node
            else:
                if node.right is None:
                    return node
                if value > node.right.value:
                    node.right.right = self.splay_and_rebalance(value, node.right.right)
                    node = self.rotate_left(node)
                elif value < node.right.value:
                    node.right.left = self.splay_and_rebalance(value, node.right.left)
                    if node.right.left is not None:
                        node.right = self.rotate_right(node.right)
                return self.rotate_left(node) if node.right is not None else node

    n = int(fin.readline())
    x = 0
    mod = 10 ** 9 + 1
    tree = SplayTree()
    for _ in range(n):
        datas = fin.readline().split()  # type, (nums 1 or 2)
        if datas[0] == '+':
            tree.insert((int(datas[1]) + x) % mod)
        elif datas[0] == '-':
            tree.delete((int(datas[1]) + x) % mod)
        elif datas[0] == '?':
            fout.write('Found\n' if tree.find((int(datas[1]) + x) % mod) else 'Not found\n')
        elif datas[0] == 's':
            x = tree.sum((int(datas[1]) + x) % mod, (int(datas[2]) + x) % mod)
            fout.write(f'{x}\n')
