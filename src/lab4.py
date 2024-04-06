# My tasks: 3, 5, 6, 9
# Points: 1 + 1.5 + 1.5 + 2 = 6
import math

from typing import TextIO, List
from utils.decorators import fin_fout, performance
from collections import deque


@performance
@fin_fout
def lab4_task3(fin: TextIO, fout: TextIO):
    def rabin_karp(pattern, st) -> List[str]:
        def start_hash(s) -> int:
            h = 0
            for char in s:
                h = (h * x + ord(char)) % mod
            return h

        x = 256
        mod = 10**9 + 7

        pattern_hash = start_hash(pattern)
        stw_hash = start_hash(st[:len(pattern)])
        occurrences = []

        for i in range(len(st) - len(pattern) + 1):
            if pattern_hash == stw_hash:
                if pattern == st[i:len(pattern) + i]:
                    occurrences.append(str(i + 1))
            if i < len(st) - len(pattern):
                stw_hash = (stw_hash * x - ord(st[i]) * pow(x, len(pattern), mod) + ord(st[i + len(pattern)])) % mod
        return occurrences

    ans = rabin_karp(*[i.strip() for i in fin.readlines()])
    fout.write(f'{len(ans)}\n')
    fout.write(f'{" ".join(ans)}')


# @performance
# @fin_fout
# def lab4_task4(fin: TextIO, fout: TextIO):
#     mod = 10**9 + 7
#     x = 31
#     # not right


@performance
@fin_fout
def lab4_task5(fin: TextIO, fout: TextIO):
    def prefix_function(s: str) -> List[int]:
        result = [0] * len(s)
        for i in range(1, len(s)):
            curr = result[i - 1]
            while s[i] != s[curr] and curr > 0:
                curr = result[curr - 1]
            if s[i] == s[curr]:
                result[i] = curr + 1
        return result

    fout.write(' '.join(list(map(str, prefix_function(fin.readline())))))


@performance
@fin_fout
def lab4_task6(fin: TextIO, fout: TextIO):
    def z_func(s: str) -> List[int]:
        result = [0] * len(s)
        l, r = 0, 0
        for i in range(1, len(s)):
            if i <= r:
                result[i] = min(r - i + 1, result[i - l])
            while i + result[i] < len(s) and s[result[i]] == s[i + result[i]]:
                result[i] += 1
            if i + result[i] - 1 > r:
                r = i + result[i] - 1
                l = i
        return result

    fout.write(' '.join(list(map(str, z_func(fin.readline())))[1:]))


# @performance
# @fin_fout
# def lab4_task7(fin: TextIO, fout: TextIO):
#     def string_hashes(s: str) -> tuple:
#         x = 13
#         mod = 10 ** 9 + 7
#         hashes = [0]
#         x_pow = [1]
#         for c in s:
#             hashes.append((hashes[-1] * x + ord(c)) % mod)
#             x_pow.append(x_pow[-1] * x)
#         return hashes, x_pow
#
#     def is_sub(hash_table, t, l):
#         hashes_t, pow_t = string_hashes(t)
#         for i in range(len(t) - l):
#             if (hashes_t[i + l] - pow_t[l] * hashes_t[i]) in hash_table[l]:
#                 return i
#         return None
#
#     def find_max_substring(s: str, t: str) -> tuple:
#         hashes_s, pow_s = string_hashes(s)
#         hash_table_s = [[(hashes_s[j + i] - pow_s[i] * hashes_s[j]) for j in range(len(s) - i + 1)]
#                         for i in range(len(s) + 1)]
#         l, r = 0, min(len(s1), len(s2))
#         mx_sub = -1
#         while l + 1 < r:  # length binary search
#             m = (l + r) // 2
#             substr = is_sub(hash_table_s, t, m)
#             if substr is not None:
#                 mx_sub = max(mx_sub, substr)
#                 l = m
#             else:
#                 r = m
#
#         if mx_sub > 0:
#             for i in range(len(s1) - l + 1):
#                 if s2[mx_sub:mx_sub + l] == s1[i: i + l]:
#                     return i, mx_sub, l
#         return 0, 0, 0
#
#     for s1, s2 in [i.strip().split() for i in fin.readlines()]:
#         fout.write(f'{" ". join(map(str, find_max_substring(s1, s2)))}\n')


@performance
@fin_fout
def lab4_task9(fin: TextIO, fout: TextIO):
    def prefix_function(st: str, offset: int) -> List[int]:
        p = [0] * (len(st) - offset)
        stackq = deque()
        stackq.append((1, 0))

        while stackq:
            i, j = stackq.pop()
            if (offset + i) < len(st):
                if st[offset + i] == st[offset + j]:
                    p[i] = j + 1
                    stackq.append((i + 1, j + 1))
                elif j > 0:
                    stackq.append((i, p[j - 1]))
                else:
                    p[i] = 0
                    stackq.append((i + 1, 0))
            else:
                return p

    def find_repeat(st: str, offset: int) -> tuple:
        p = prefix_function(st, offset)
        stackq = deque()
        stackq.append((1, p[0], 0))
        m, mi = 0, 0

        while stackq:
            i, mx, mindx = stackq.pop()
            if i < (len(st) - offset):
                if (p[i] > mx) and ((i + 1) % (i + 1 - p[i]) == 0):
                    stackq.append((i + 1, p[i], i))
                else:
                    stackq.append((i + 1, mx, mindx))
            else:
                m, mi = mx, mindx
                break

        if m > 0:
            return (mi - m + 1), ((mi + 1) // (mi + 1 - m))
        else:
            return 1, 1

    def decomposition_string(st: str) -> str:
        l1 = [0] * (len(st) + 1)
        l2 = [10000] * (len(st) + 1)
        p = [(0, 0)] * len(st)

        for i in range(len(st) - 1, -1, -1):
            l1[i] = 1 + min(l1[i + 1], 1 + l2[i + 1])
            l, k = find_repeat(st, i)
            p[i] = (l, k)
            msl = min(l1[i + l * k], l2[i + l * k])
            l2[i] = l + 1 + int(1 + math.log10(k)) + (msl > 0) + msl

        result_string = ''
        stackq = deque()
        stackq.append((0, False))

        while stackq:
            curr, add_minus = stackq.pop()
            if curr < len(st):
                if l1[curr] <= l2[curr]:
                    if add_minus:
                        result_string += '+'

                    result_string += st[curr]
                    stackq.append((curr + 1, False))
                else:
                    l, k = p[curr]
                    if curr > 0:
                        result_string += '+'
                    result_string += st[curr:curr + l] + '*' + str(k)
                    stackq.append((curr + l * k, True))

        return result_string

    fout.write(decomposition_string(fin.readline()))
