# My tasks: 3, 4, 8, 9, 10, 11, 14, 17, 18, 19, 20, 21, 22
# Points: 0.5 + 0.5 + 1 + 1 + 1 + 1 + 2 + 2.5 + 2.5 + 3 + 3 + 3 + 4 = 25
from typing import TextIO
from utils.decorators import fin_fout, performance
from math import ceil
from collections import deque


@performance
@fin_fout
def lab1_task3(fin: TextIO, fout: TextIO):
    n = int(fin.readline())
    costs = sorted(list(map(int, fin.readline().split())))
    clicks = sorted(list(map(int, fin.readline().split())))
    result = 0
    for i in range(0, n):
        result += costs[i] * clicks[i]
    fout.write(str(result))


@performance
@fin_fout
def lab1_task4(fin: TextIO, fout: TextIO):
    n = int(fin.readline())
    points = [list(map(int, fin.readline().split())) for _ in range(n)]
    points = sorted(points, key=lambda x: x[1])
    curr_end = 0
    ends = []
    for point in points:
        if curr_end < point[0]:
            curr_end = point[1]
            ends.append(str(point[1]))
    fout.write(f'{str(len(ends))}\n{" ".join(ends)}')


@performance
@fin_fout
def lab1_task8(fin: TextIO, fout: TextIO):
    n = int(fin.readline())
    lectures = sorted([[int(i) for i in fin.readline().split()] for _ in range(n)], key=lambda x: x[1])
    last = 0
    c = 0
    for lecture in lectures:
        if last <= lecture[0]:
            c += 1
            last = lecture[1]
    fout.write(str(c))


@performance
@fin_fout
def lab1_task9(fin: TextIO, fout: TextIO):
    class InfoLists:
        def __init__(self, cost, counts):
            self.cost = cost
            self.counts = counts

        def price_for_n(self, lists):
            return self.cost * ceil(lists / self.counts)

    n = int(fin.readline())
    costs = [InfoLists(int(fin.readline()), 10 ** i) for i in range(7)]
    summ = 0
    while n > 0:
        curr_n = n // (10 ** (len(str(n)) - 1)) * (10 ** (len(str(n)) - 1))
        curr_costs = [cost.price_for_n(curr_n) for cost in costs]
        mc = min(curr_costs)
        for i in range(6, -1, -1):
            if curr_costs[i] == mc:
                summ += mc
                n -= max(curr_n, costs[i].counts)
                break
    fout.write(str(summ))


@performance
@fin_fout
def lab1_task10(fin: TextIO, fout: TextIO):
    class Apple:
        def __init__(self, a, b, ind):
            self.negative = a
            self.positive = b
            self.index = ind + 1

        def __repr__(self):
            return f'Apple(negative = {self.negative}, positive = {self.positive}, index = {self.index})'

    n, height = list(map(int, fin.readline().split()))
    positive_apples = []
    negative_apples = []
    for i in range(n):
        param = Apple(*list(map(int, fin.readline().split())), i)
        if (param.positive - param.negative) >= 0:
            positive_apples.append(param)
        else:
            negative_apples.append(param)
    positive_apples.sort(key=lambda x: x.negative)
    negative_apples.sort(key=lambda x: x.negative)
    all_apples: list[Apple] = [*positive_apples, *negative_apples]
    story = []
    for apple in all_apples:
        height -= apple.negative
        if height <= 0:
            break
        height += apple.positive
        story.append(str(apple.index))
    if height <= 0:
        fout.write('-1')
    else:
        fout.write(' '.join(story))


@performance
@fin_fout
def lab1_task11(fin: TextIO, fout: TextIO):
    w, n = list(map(int, fin.readline().split()))
    ingots = sorted(list(map(int, fin.readline().split())), reverse=True)
    result = 0
    for ingot in ingots:
        if ingot <= w:
            w -= ingot
            result += ingot
    fout.write(str(result))


@performance
@fin_fout
def lab1_task14(fin: TextIO, fout: TextIO):
    def max_value(expression: str) -> int:
        nums = []
        operas = []
        num = 0
        for ch in expression:
            if ch == '+' or ch == '-' or ch == '*':
                nums.append(num)
                operas.append(ch)
                num = 0
            else:
                num = num * 10 + int(ch)

        nums.append(num)
        mx = [[-10**10] * len(nums) for _ in range(len(nums))]
        mn = [[10**10] * len(nums) for _ in range(len(nums))]

        for i in range(len(nums)):
            mx[i][i] = nums[i]
            mn[i][i] = nums[i]

        for i in range(2, len(nums) + 1):
            for j in range(len(nums) - i + 1):
                k = j + i - 1
                max_val = - 10 ** 10
                min_val = 10 ** 10
                for l in range(j, k):
                    op = operas[l]
                    if op == '+':
                        max_val = max(max_val, mx[j][l] + mx[l + 1][k])
                        min_val = min(min_val, mn[j][l] + mn[l + 1][k])
                    elif op == '-':
                        max_val = max(max_val, mx[j][l] - mn[l + 1][k])
                        min_val = min(min_val, mn[j][l] - mx[l + 1][k])
                    elif op == '*':
                        max_val = max(max_val, mx[j][l] * mx[l + 1][k])
                        min_val = min(min_val, mn[j][l] * mn[l + 1][k])
                mx[j][k] = max_val
                mn[j][k] = min_val
        return mx[0][len(nums) - 1]

    expression = fin.readline()
    fout.write(str(max_value(expression)))


@performance
@fin_fout
def lab1_task17(fin: TextIO, fout: TextIO):
    n = int(fin.readline())
    nums = [[1] * 10 if i == 0 else [0] * 10 for i in range(n)]
    nums[0][0], nums[0][8] = 0, 0
    for i in range(1, n):
        nums[i][0] = nums[i-1][4] + nums[i-1][6]
        nums[i][1] = nums[i-1][6] + nums[i-1][8]
        nums[i][2] = nums[i-1][7] + nums[i-1][9]
        nums[i][3] = nums[i-1][4] + nums[i-1][8]
        nums[i][4] = nums[i-1][0] + nums[i-1][3] + nums[i-1][9]
        nums[i][6] = nums[i-1][0] + nums[i-1][1] + nums[i-1][7]
        nums[i][7] = nums[i-1][2] + nums[i-1][6]
        nums[i][8] = nums[i-1][1] + nums[i-1][3]
        nums[i][9] = nums[i-1][2] + nums[i-1][4]
    fout.write(str(sum(nums[n - 1]) % (10**9)))


@performance
@fin_fout
def lab1_task18(fin: TextIO, fout: TextIO):
    def calculate_day_costs():
        nonlocal coup_sp
        for i in range(n):
            if costs[i] > 100:
                for j in range(n):
                    if j == 0:
                        coup_sp[i + 1][j + 1] = min(coup_sp[i + 1][j + 1], coup_sp[i][j] + costs[i])
                    else:
                        coup_sp[i + 1][j - 1] = min(coup_sp[i + 1][j - 1], coup_sp[i][j])
                        coup_sp[i + 1][j + 1] = min(coup_sp[i + 1][j + 1], coup_sp[i][j] + costs[i])
            else:
                for j in range(n):
                    if j == 0:
                        coup_sp[i + 1][j] = min(coup_sp[i + 1][j], coup_sp[i][j] + costs[i])
                    else:
                        coup_sp[i + 1][j - 1] = min(coup_sp[i + 1][j - 1], coup_sp[i][j])
                        coup_sp[i + 1][j] = min(coup_sp[i + 1][j], coup_sp[i][j] + costs[i])

    n = int(fin.readline())
    costs = [int(fin.readline()) for _ in range(n)]
    coup_sp = [[10**9] * (n + 1) for _ in range(n + 1)]
    coup_sp[0][0] = 0
    calculate_day_costs()
    coupons = []
    count_coupons = 0
    curr_min = min(coup_sp[-1])
    for i in range(n):
        if coup_sp[-1][i] == curr_min:
            count_coupons = i
    for i in range(n, -1, -1):
        for j in range(n):
            if coup_sp[i][j] == curr_min:
                for k in range(n):
                    if coup_sp[i - 1][k] == coup_sp[i][j]:
                        coupons.append(i)
                        curr_min = coup_sp[i - 1][k]
                    elif abs(coup_sp[i - 1][k] - coup_sp[i][j]) == costs[i - 1]:
                        curr_min = coup_sp[i - 1][k]
    ans = '\n'.join(map(str, reversed(coupons)))
    ans = f'{min(coup_sp[-1])}\n{count_coupons} {len(coupons)}\n{ans}\n'
    fout.write(ans)


@performance
@fin_fout
def lab1_task19(fin: TextIO, fout: TextIO):
    def dp_matrix(matrices: list, n: int) -> list:
        count_opera = [[0] * n for _ in range(n)]
        result = [[0] * n for _ in range(n)]
        for i in range(1, n):
            for j in range(0, n - i):
                k = j + i
                count_opera[j][k] = 10**10
                for l in range(j, i):
                    num = count_opera[j][l] + count_opera[l + 1][k] + matrices[j] * matrices[l + 1] * matrices[k + 1]
                    if num < count_opera[j][k]:
                        count_opera[j][k] = num
                        result[j][k] = l
        return result

    def get_ans(i: int, j: int) -> str:
        nonlocal res
        ans = ''
        deq = deque()
        deq.append((i, j, 'o'))  # i, j, type : 'o' - open, 'c' - close
        while deq:
            l1, l2, typ = deq.popleft()
            if l1 == l2:
                ans += 'A'
            else:
                if typ == 'o':
                    ans += '('
                    deq.appendleft((l1, l2, 'c'))
                    deq.appendleft((res[l1][l2] + 1, l2, 'o'))
                    deq.appendleft((l1, res[l1][l2], 'o'))
                elif typ == 'c':
                    ans += ')'
        return ans

    n = int(fin.readline())
    nums = []
    num1, num2 = 0, 0
    for i in range(n):
        num1, num2 = map(int, fin.readline().split())
        nums.append(num1)
    nums.append(num2)
    res = dp_matrix(nums, n)
    fout.write(get_ans(0, len(nums) - 2))



@performance
@fin_fout
def lab1_task20(fin: TextIO, fout: TextIO):
    n, k = list(map(int, fin.readline().split()))
    string = str(fin.readline())
    count = 0
    for i in range(0, len(string)):
        offset = 0
        curr_k = k
        while (i - offset) >= 0 and (i + offset) < len(string) and curr_k >= 0:
            if string[i - offset] == string[i + offset]:
                count += 1
                offset += 1
            else:
                curr_k -= 1
                offset += 1
                if curr_k >= 0:
                    count += 1
    for i in range(0, len(string) - 1):
        offset = 0
        curr_k = k
        while (i - offset) >= 0 and (i + 1 + offset) < len(string) and curr_k >= 0:
            if string[i - offset] == string[i + 1 + offset]:
                count += 1
                offset += 1
            else:
                curr_k -= 1
                offset += 1
                if curr_k >= 0:
                    count += 1
    fout.write(str(count))


@performance
@fin_fout
def lab1_task21(fin: TextIO, fout: TextIO):
    def sorted_card(cards: list) -> tuple:
        s_cards = []
        c_cards = []
        d_cards = []
        h_cards = []
        for card in cards:
            if card[-1] == 'S':
                s_cards.append(card)
            elif card[-1] == 'C':
                c_cards.append(card)
            elif card[-1] == 'D':
                d_cards.append(card)
            else:
                h_cards.append(card)

        def comparator(s):
            nonlocal order
            return order[s[:-1]]

        return (
            sorted(s_cards, key=comparator),
            sorted(c_cards, key=comparator),
            sorted(d_cards, key=comparator),
            sorted(h_cards, key=comparator)
        )

    order = {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4, 'T': 5, 'J': 6, 'Q': 7, 'K': 8, 'A': 9}
    trump = fin.readline().split()[-1]
    mys, myc, myd, myh = sorted_card(fin.readline().split())
    es, ec, ed, eh = sorted_card(fin.readline().split())
    my_trump, e_trump = [], []
    if trump == 'S':
        my_trump, e_trump = mys, es
        mys, es = [], []
    elif trump == 'C':
        my_trump, e_trump = myc, ec
        myc, ec = [], []
    elif trump == 'D':
        my_trump, e_trump = myd, ed
        myd, ed = [], []
    elif trump == 'H':
        my_trump, e_trump = myh, eh
        myh, eh = [], []
    if len(e_trump) > len(my_trump):
        fout.write('NO')
        return
    while e_trump:
        card = e_trump.pop()
        my_card = my_trump.pop()
        if order[card[:-1]] >= order[my_card[:-1]]:
            fout.write('NO')
            return
    while es:
        card = es.pop()
        if len(mys) != 0:
            if order[mys[-1][:-1]] > order[card[:-1]]:
                mys.pop()
            else:
                if len(my_trump) == 0:
                    fout.write('NO')
                    return
                else:
                    my_trump.pop()
        else:
            if len(my_trump) == 0:
                fout.write('NO')
                return
            else:
                my_trump.pop()
    while ec:
        card = ec.pop()
        if len(myc) != 0:
            if order[myc[-1][:-1]] > order[card[:-1]]:
                myc.pop()
            else:
                if len(my_trump) == 0:
                    fout.write('NO')
                    return
                else:
                    my_trump.pop()
        else:
            if len(my_trump) == 0:
                fout.write('NO')
                return
            else:
                my_trump.pop()
    while ed:
        card = ed.pop()
        if len(myd) != 0:
            if order[myd[-1][:-1]] > order[card[:-1]]:
                myd.pop()
            else:
                if len(my_trump) == 0:
                    fout.write('NO')
                    return
                else:
                    my_trump.pop()
        else:
            if len(my_trump) == 0:
                fout.write('NO')
                return
            else:
                my_trump.pop()
    while eh:
        card = eh.pop()
        if len(myh) != 0:
            if order[myh[-1][:-1]] > order[card[:-1]]:
                myh.pop()
            else:
                if len(my_trump) == 0:
                    fout.write('NO')
                    return
                else:
                    my_trump.pop()
        else:
            if len(my_trump) == 0:
                fout.write('NO')
                return
            else:
                my_trump.pop()
    fout.write('YES')
