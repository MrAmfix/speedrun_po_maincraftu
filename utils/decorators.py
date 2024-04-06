import time
import memory_profiler


def fin_fout(func):
    def wrapper(*args, **kwargs):
        path_in = f'input/{func.__name__.split("_")[0]}/{func.__name__.split("_")[1]}.txt'
        path_out = f'output/{func.__name__.split("_")[0]}/{func.__name__.split("_")[1]}.txt'
        with open(path_in, 'r', encoding='utf-8') as fin, open(path_out, 'w', encoding='utf-8') as fout:
            return func(fin, fout, *args, **kwargs)
    return wrapper


def performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_mem = memory_profiler.memory_usage()[0]
        res = func(*args, **kwargs)
        end_time = time.time()
        end_mem = memory_profiler.memory_usage()[0]
        print(f'Time: {(end_time - start_time) * 1000} ms')
        print(f'Memory: {(end_mem - start_mem) * 1024} KB')
        return res
    return wrapper
