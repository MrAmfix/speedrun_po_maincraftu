
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.height = 1

    def __repr__(self):
        return str(self.value)


class AVLTree:
    def insert(self, node, value):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self.insert(node.left, value)
        else:
            node.right = self.insert(node.right, value)

        node.height = 1 + max(self.get_height(node.left),
                              self.get_height(node.right))
        return self.rebalance(node, value)

    def delete(self, node, value):
        if not node:
            return node
        elif value < node.value:
            node.left = self.delete(node.left, value)
        elif value > node.value:
            node.right = self.delete(node.right, value)
        else:
            if node.left is None:
                temp = node.right
                return temp
            elif node.right is None:
                temp = node.left
                return temp
            temp = self.min_from_node(node.right)
            node.value = temp.value
            node.right = self.delete(node.right, temp.value)
        if node is None:
            return node

        node.height = 1 + max(self.get_height(node.left),
                              self.get_height(node.right))

        return self.rebalance(node, value)


    def rebalance(self, node, value):
        balance = self.get_balance(node)

        if balance >= 2:
            if value < node.left.value:
                return self.right_rotate(node)
            else:
                node.left = self.left_rotate(node.left)
                return self.right_rotate(node)
        if balance <= -2:
            if value > node.right.value:
                return self.left_rotate(node)
            else:
                node.right = self.right_rotate(node.right)
                return self.left_rotate(node)
        return node

    def find_elem(self, node, value):
        while node is not None:
            if node.value == value:
                return True
            elif node.value < value:
                node = node.right
            elif node.value > value:
                node = node.left
        return False

    def next(self, node, value, subfind=None):
        if node is None:
            return subfind
        if node.value > value:
            return self.next(node.left, value, node.value)
        elif node.value <= value:
            return self.next(node.right, value, subfind)

    def prev(self, node, value, subfind=None):
        if node is None:
            return subfind
        if node.value < value:
            return self.prev(node.right, value, node.value)
        elif node.value >= value:
            return self.prev(node.left, value, subfind)

    def left_rotate(self, node):
        snode = node.right
        snode2 = snode.left
        snode.left = node
        node.right = snode2
        node.height = 1 + max(self.get_height(node.left),
                              self.get_height(node.right))
        snode.height = 1 + max(self.get_height(snode.left),
                               self.get_height(snode.right))
        return snode


    def right_rotate(self, node):
        snode = node.left
        snode2 = snode.right
        snode.right = node
        node.left = snode2
        node.height = 1 + max(self.get_height(node.left),
                              self.get_height(node.right))
        snode.height = 1 + max(self.get_height(snode.left),
                               self.get_height(snode.right))
        return snode

    def get_height(self, node):
        if not node:
            return 0
        return node.height

    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def min_from_node(self, node):
        if node is None or node.left is None:
            return node
        return self.min_from_node(node.left)

    def max_from_node(self, node):
        if node is None or node.right is None:
            return node
        return self.max_from_node(node.right)

    def k_max(self, node, k, visited: set):
        if node is None:
            return None
        right_val = self.k_max(node.right, k, visited)
        if right_val is not None:
            return right_val
        visited.add(node.value)
        if len(visited) == k:
            return node.value
        left_val = self.k_max(node.left, k, visited)
        if left_val is not None:
            return left_val
        return None
