import datetime
import csv
import math
import random
import numpy as np


def func_rand(time:float):
    # """Генерирует случайное значение для столбца func1"""
    res = 2500 + random.random() * 1000
    return res

def func_sin(time:float, period:int = 220000, v1:int = 2500, v2:float = 1000, v3:float = 210.0):
    # """Синусоидальная функция с шумом для столбца func2"""
    s = math.sin((2 * np.pi / period) * time)
    res = v1 + s * v2 + random.random() * v3
    return res

def func_three(time:float, period:int = 220000, v1:int = 2500, v2:float = 1000, v3:float = 210.0):
    # """Комбинация func_sin и func_rand для столбца func3"""
    res = func_sin(time, period, v1, v2, v3) + func_rand(time)
    return res

def func_four(time:float, period:int = 220000, v1:int = 2500, v2:float = 1000, v3:float = 210.0):
    # """time + func_three + случайный шум — данные для столбца func4"""
    res = time + func_three(time, period, v1, v2, v3) + func_rand(time)
    return res


def generate_csv(filename:str, start:int, count:int, period:int, v1:int, v2:float, v3:float):
    # """Генерирует CSV-файл с пятью функциями по заданным параметрам"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['time', 'func1', 'func2', 'func3', 'func4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(count):
            writer.writerow({
                'time': start + i,
                'func1': func_rand(start + i),
                'func2': func_sin(start + i , period, v1, v2, v3),
                'func3': func_three(start + i , period, v1, v2, v3),
                'func4': func_four(start + i, period, v1, v2, v3)
            })


if __name__ == '__main__':
    start = int(datetime.datetime.now().timestamp())

    # параметры варианта 6
    period = 220000
    v1 = 2500
    v2 = 1000
    v3 = 210.0

    generate_csv('test.csv', start, 10 ** 6, period, v1, v2, v3)
    print("CSV-файл 'test.csv' успешно сгенерирован.")