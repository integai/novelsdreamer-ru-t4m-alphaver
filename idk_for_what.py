import math
import random

def calculate_array_base(x):
    array_base = []
    for i in range(100):
        if x != 0:
            x_value = (math.cos(x) / math.sin(x)) * math.pi
            array_base.append(x_value)
            x *= random.uniform(0.1, 100.0)
        else:
            array_base.append(0)
    return array_base

x = float(input('Write X: '))
x_array = calculate_array_base(x)
print(x_array)
