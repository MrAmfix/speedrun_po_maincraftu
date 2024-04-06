# My tasks: 1, 2, 3, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18
# Points: 1 + 1 + 1 + 1.5 + 1.5 + 2 + 3 + 3 + 3 + 3 + 3 + 4 + 4 = 31
import typing
import heapq

from collections import deque
from sys import setrecursionlimit
from typing import TextIO, List, Dict
from utils.decorators import fin_fout, performance
from math import sqrt


@performance
@fin_fout
def lab3_task1(fin: TextIO, fout: TextIO):
    def bfs() -> str:
        nonlocal f, t, edges
        deq = deque()
        visited = []
        deq.append(f)
        while deq:
            curr_v = deq.popleft()
            visited.append(curr_v)
            if curr_v == t:
                return '1'
            if curr_v in edges:
                for v in edges[curr_v]:
                    if v not in visited:
                        deq.append(v)
        return '0'

    nv, ne = map(int, fin.readline().split())
    edges: Dict[int, List[int]] = {}
    for _ in range(ne):
        s, e = map(int, fin.readline().split())
        edges.setdefault(s, []).append(e)
        edges.setdefault(e, []).append(s)
    f, t = map(int, fin.readline().split())
    fout.write(bfs())


@performance
@fin_fout
def lab3_task2(fin: TextIO, fout: TextIO):
    def bfs(v: int) -> int:
        nonlocal edges, visited
        if v in visited:
            return 0
        deq = deque()
        deq.append(v)
        while deq:
            curr_v = deq.popleft()
            visited.append(curr_v)
            if curr_v in edges:
                for v in edges[curr_v]:
                    if v not in visited:
                        deq.append(v)
        return 1


    nv, ne = map(int, fin.readline().split())
    edges: Dict[int, List[int]] = {}
    for _ in range(ne):
        s, e = map(int, fin.readline().split())
        edges.setdefault(s, []).append(e)
        edges.setdefault(e, []).append(s)

    visited = []
    summ = 0
    for i in range(1, nv + 1):
        summ += bfs(i)
    fout.write(str(summ))


@performance
@fin_fout
def lab3_task3(fin: TextIO, fout: TextIO):
    setrecursionlimit(2000)

    class Vertex:
        def __init__(self):
            self.neighbours = []

        def add_neighbours(self, neighbour):
            self.neighbours.append(neighbour)

    def dfs(vert: int, visit_story: set) -> bool:
        if vert in not_visited:
            not_visited.remove(vert)
        visit_story.add(vert)
        for neighbour in vertices[vert].neighbours:
            if neighbour in visit_story:
                return True
            elif neighbour not in not_visited:
                pass
            else:
                flag = dfs(neighbour, visit_story.copy())
                if flag:
                    return True
        return False

    nv, ne = list(map(int, fin.readline().split()))
    vertices = [Vertex() for _ in range(nv)]
    for _ in range(ne):
        f, t = list(map(int, fin.readline().split()))
        vertices[f-1].add_neighbours(t-1)
    not_visited = set([i for i in range(nv)])
    is_cycle = False
    while len(not_visited) > 0 and not is_cycle:
        start_vertex = not_visited.pop()
        is_cycle = dfs(start_vertex, set())
    if is_cycle:
        fout.write('1')
    else:
        fout.write('0')


@performance
@fin_fout
def lab3_task7(fin: TextIO, fout: TextIO):
    class Vertex:
        def __init__(self):
            self.neighbours = set()

        def add_neighbour(self, neighbour):
            self.neighbours.add(neighbour)

    def dfs(vert: int, list_vertices) -> bool:
        stackq = deque()
        stackq.append((vert, None))
        while stackq:
            c_vert, c_prev = stackq.pop()
            visited.add(c_vert)
            for neighbour in list_vertices[c_vert].neighbours:
                if neighbour in visited and neighbour != c_prev and c_prev is not None:
                    return True
                elif neighbour not in visited:
                    stackq.append((neighbour, c_vert))
        return False

    nv, ne = list(map(int, fin.readline().split()))
    vertices = [Vertex() for _ in range(nv)]
    for _ in range(ne):
        f, t = list(map(int, fin.readline().split()))
        vertices[f - 1].add_neighbour(t - 1)
        vertices[t - 1].add_neighbour(f - 1)
    visited = set()
    is_cycle = dfs(0, vertices)
    if is_cycle:
        fout.write('0')
    else:
        fout.write('1')


@performance
@fin_fout
def lab3_task8(fin: TextIO, fout: TextIO):
    class Flight:
        def __init__(self, end_place, cost):
            self.end_place = end_place
            self.cost = cost

    def dijkstra(n, graph, start_place) -> list:
        costs = [float('inf')] * n
        costs[start_place] = 0
        queue = [(0, start_place)]

        while queue:
            curr_cost, current_node = heapq.heappop(queue)
            if curr_cost > costs[current_node]:
                continue
            for flight in graph[current_node]:
                new_cost = curr_cost + flight.cost
                if new_cost < costs[flight.end_place]:
                    costs[flight.end_place] = new_cost
                    heapq.heappush(queue, (new_cost, flight.end_place))
        return costs

    nv, ne = map(int, fin.readline().split())
    graph = {i: [] for i in range(nv)}

    for _ in range(ne):
        f, t, c = map(int, fin.readline().split())
        graph[f - 1].append(Flight(t - 1, c))

    st, to = map(int, fin.readline().split())
    min_cost = dijkstra(nv, graph, st - 1)[to - 1]
    fout.write(str(min_cost) if min_cost != float('inf') else '-1')


@performance
@fin_fout
def lab3_task9(fin: TextIO, fout: TextIO):
    def bellman_ford() -> str:
        nonlocal graph, n
        distances = [10 ** 10] * n
        distances[0] = 0
        for _ in range(n - 1):
            for f, t, c in graph:
                if distances[f] != 10 ** 10 and distances[f] + c < distances[t]:
                    distances[t] = distances[f] + c
        for f, t, c in graph:
            if distances[f] != 10 ** 10 and distances[f] + c < distances[t]:
                return '1'
        return '0'

    n, m = map(int, fin.readline().split())
    graph = []
    for _ in range(m):
        f, t, c = map(int, fin.readline().split())
        graph.append((f - 1, t - 1, c))
    fout.write(bellman_ford())


@performance
@fin_fout
def lab3_task11(fin, fout):
    def bfs(start_v, end_v):
        visited_v = set()
        deq = deque([(start_v, 0)])
        while deq:
            curr_v, curr_c = deq.popleft()
            visited_v.add(curr_v)
            if curr_v == end_v:
                return curr_c
            curr_c += 1
            if curr_v in reactions:
                for v in reactions[curr_v]:
                    if v not in visited_v:
                        deq.append((v, curr_c))
        return -1

    n = int(fin.readline())
    reactions = {}
    for _ in range(n):
        f, t = fin.readline().strip().split(' -> ')
        reactions.setdefault(f, []).append(t)
    fout.write(f'{bfs(fin.readline().strip(), fin.readline().strip())}')


@performance
@fin_fout
def lab3_task13(fin: TextIO, fout: TextIO):
    def sub_func():
        nonlocal coords
        for i, j in coords:
            bfs(i, j)

    def bfs(i, j): # pos from
        nonlocal result, graph, n, m
        if graph[i][j] == '#':  # если до этого не смотрели
            result += 1
            deq = deque()
            deq.append((i, j))
            while deq:
                i, j = deq.popleft() # i (n), j (m)  (5, 10)
                graph[i][j] = '.'
                if i > 0:
                    if graph[i - 1][j] == '#':
                        deq.append((i - 1, j))
                if i < (n - 1):
                    if graph[i + 1][j] == '#':
                        deq.append((i + 1, j))
                if j > 0:
                    if graph[i][j - 1] == '#':
                        deq.append((i, j - 1))
                if j < (m - 1):
                    if graph[i][j + 1] == '#':
                        deq.append((i, j + 1))

    n, m = map(int, fin.readline().split())  # 5 10 - example
    graph = []  # i (n), j (m)
    result = 0
    coords = []
    for i in range(n):
        s = list(fin.readline().strip())
        graph.append(s)
        for j in range(len(s)):
            if s[j] == '#':
                coords.append((i, j))
    sub_func()
    fout.write(f'{result}')


@performance
@fin_fout
def lab3_task14(fin: TextIO, fout: TextIO):
    class Flight:
        def __init__(self, start_time, end_place, end_time):
            self.start_time = start_time
            self.end_place = end_place
            self.end_time = end_time

    def dijkstra(flights: dict, d: int, v: int) -> int:
        nonlocal n
        distances = [10 ** 10 for _ in range(n + 1)]
        distances[d] = 0
        curr_place = d
        visited = []
        flag = True
        while flag:
            if curr_place == v:
                return distances[v]
            visited.append(curr_place)
            if curr_place in flights:
                for flight in flights[curr_place]:
                    if flight.end_place not in visited and flight.start_time >= distances[curr_place]:
                        distances[flight.end_place] = min(distances[flight.end_place], flight.end_time)
            min_time = 10 ** 10
            for i in range(1, len(distances)):
                if i not in visited and distances[i] < min_time:
                    min_time = distances[i]
                    curr_place = i
            if min_time == 10 ** 10:
                flag = False
        return distances[v] if distances[v] != 10 ** 10 else -1

    n = int(fin.readline())
    d, v = map(int, fin.readline().split())
    r = int(fin.readline())
    flights: Dict[int, List[Flight]] = {}  # Dict with key = number of place, value - list of flights
    for _ in range(r):
        from_p, from_t, to_p, to_t = map(int, fin.readline().split())
        if from_p in flights:
            flights[from_p].append(Flight(from_t, to_p, to_t))
        else:
            flights[from_p] = [Flight(from_t, to_p, to_t)]
    fout.write(str(dijkstra(flights, d, v)))


@performance
@fin_fout
def lab3_task15(fin: TextIO, fout: TextIO):
    class People:
        def __init__(self, x, y, drag):
            self.x = x - 1
            self.y = y - 1
            self.drag = drag  # У королевы drag - это время

    def bfs(graph: List[List[int]], queen, atos, portos, aramis, dartanian):
        # Тут поменяли оси (вертикаль x, m ; горизонталь y, n)
        nonlocal n, m
        graph[queen.x][queen.y] = 0
        deq = deque()
        deq.append((queen.x, queen.y))

        while deq:
            curr_x, curr_y = deq.popleft()

            if curr_x < (m - 1):  # Можем идти вниз
                if graph[curr_x + 1][curr_y] != 10 ** 10:
                    if 1 + graph[curr_x][curr_y] < graph[curr_x + 1][curr_y]:
                        graph[curr_x + 1][curr_y] = 1 + graph[curr_x][curr_y]
                        deq.append((curr_x + 1, curr_y))
            if curr_x > 0:  # Можем идти вверх
                if graph[curr_x - 1][curr_y] != 10 ** 10:
                    if 1 + graph[curr_x][curr_y] < graph[curr_x - 1][curr_y]:
                        graph[curr_x - 1][curr_y] = 1 + graph[curr_x][curr_y]
                        deq.append((curr_x - 1, curr_y))
            if curr_y > 0:  # Можем идти влево
                if graph[curr_x][curr_y - 1] != 10 ** 10:
                    if 1 + graph[curr_x][curr_y] < graph[curr_x][curr_y - 1]:
                        graph[curr_x][curr_y - 1] = 1 + graph[curr_x][curr_y]
                        deq.append((curr_x, curr_y - 1))
            if curr_y < (n - 1):  # Можем идти вправо
                if graph[curr_x][curr_y + 1] != 10 ** 10:
                    if 1 + graph[curr_x][curr_y] < graph[curr_x][curr_y + 1]:
                        graph[curr_x][curr_y + 1] = 1 + graph[curr_x][curr_y]
                        deq.append((curr_x, curr_y + 1))

        # for i in range(n):
        #     for j in range(m):
        #         print(f'{graph[i][j]} ', end='')
        #     print('\n', end='')
        sum_drag = 0
        if graph[atos.x][atos.y] <= queen.drag:
            sum_drag += atos.drag
        if graph[portos.x][portos.y] <= queen.drag:
            sum_drag += portos.drag
        if graph[aramis.x][aramis.y] <= queen.drag:
            sum_drag += aramis.drag
        if graph[dartanian.x][dartanian.y] <= queen.drag:
            sum_drag += dartanian.drag
        return sum_drag

    n, m = map(int, fin.readline().split())
    # Вершина по координатам i, j обладает номером n * j + i
    graph = []
    for i in range(n):
        graph.append([])
        for j in fin.readline():
            if j == '1':
                graph[-1].append(10 ** 10)
            else:
                graph[-1].append(10 ** 9)
    # 10 ** 10 - туда нельзя, 10 ** 9 - туда можно
    queen = People(*list(map(int, fin.readline().split())))
    atos = People(*list(map(int, fin.readline().split())))
    portos = People(*list(map(int, fin.readline().split())))
    aramis = People(*list(map(int, fin.readline().split())))
    dartanian = People(*list(map(int, fin.readline().split())))
    fout.write(f'{bfs(graph, queen, atos, portos, aramis, dartanian)}')



@performance
@fin_fout
def lab3_task16(fin: TextIO, fout: TextIO):
    class Vertex:
        def __init__(self):
            self.neighbours = set()

        def add_neighbour(self, neighbour):
            self.neighbours.add(neighbour)

    def dfs(vert: str, list_vertices: typing.Dict[str, Vertex]):
        stackq: typing.Deque[typing.Tuple[str, str]] = deque()
        stackq.append((vert, vert))  # current_vert, start_vert
        visited = set()
        while stackq:
            c_vert, start_vert = stackq.pop()
            if c_vert not in visited:
                visited.add(c_vert)
            for neighbour in list_vertices[c_vert].neighbours:
                if neighbour == start_vert:
                    return True
                elif neighbour in visited:
                    pass
                else:
                    stackq.append((neighbour, start_vert))
        return False

    n = int(fin.readline())
    vertices = dict()
    for i in range(n):
        id_func = fin.readline()
        sub_n = int(fin.readline())
        vertices[id_func] = Vertex()
        for j in range(sub_n):
            vertices[id_func].add_neighbour(fin.readline())
        fin.readline()  # Reading *****

    for key in vertices.keys():
        fout.write('YES\n' if dfs(key, vertices) else 'NO\n')


@performance
@fin_fout
def lab3_task17(fin: TextIO, fout: TextIO):
    def floyd_warshall(graph: List[List[int]]) -> List[List[int]]:
        for i in range(len(graph)):
            for j in range(len(graph)):
                for k in range(len(graph)):
                    if graph[j][i] + graph[i][k] < graph[j][k]:
                        graph[j][k] = graph[j][i] + graph[i][k]
        return graph

    nv, ne = list(map(int, fin.readline().split()))
    graph = [[10**10] * nv for _ in range(nv)]
    for i in range(0, nv):
        graph[i][i] = 0
    for _ in range(ne):
        s, e = list(map(int, fin.readline().split()))
        graph[s - 1][e - 1] = 0
        graph[e - 1][s - 1] = min(graph[e - 1][s - 1], 1)

    ans = 0
    result = floyd_warshall(graph.copy())
    for i in result:
        for j in i:
            ans = max(ans, j)
    fout.write(str(ans))


@performance
@fin_fout
def lab3_task18(fin: TextIO, fout: TextIO):
    class Kruskal:
        def __init__(self, n, graph):
            self.n = n
            self.graph = graph

        def find(self, parent, i):
            if parent[i] == i:
                return i
            return self.find(parent, parent[i])

        def union(self, parent, rank, x, y):
            x_root = self.find(parent, x)
            y_root = self.find(parent, y)

            if rank[x_root] < rank[y_root]:
                parent[x_root] = y_root
            elif rank[x_root] > rank[y_root]:
                parent[y_root] = x_root
            else:
                parent[y_root] = x_root
                rank[x_root] += 1

        def min_cost_graph(self):
            result = []
            self.graph = sorted(self.graph, key=lambda item: item[2])
            parent = [i for i in range(self.n)]
            rank = [0] * self.n
            i, e = 0, 0
            while e < self.n - 1:
                u, v, w = self.graph[i]
                i = i + 1
                x = self.find(parent, u)
                y = self.find(parent, v)
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.union(parent, rank, x, y)

            min_cost = 0
            for u, v, weight in result:
                min_cost += weight
            return min_cost

    n = int(fin.readline())
    points = []
    for _ in range(n):
        x, y = list(map(int, fin.readline().split()))
        points.append((x, y))
    graph = []
    for i in range(n):
        for j in range(i + 1, n):
            graph.append([i, j, sqrt((points[i][0] - points[j][0]) ** 2
                                     + (points[i][1] - points[j][1]) ** 2)])
    kr = Kruskal(n, graph)
    fout.write('{:.8f}'.format(kr.min_cost_graph()))
