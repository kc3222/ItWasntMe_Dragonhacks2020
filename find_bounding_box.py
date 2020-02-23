# %%
import pandas as pd
import numpy as np
import ast
import json

# %%
df = pd.read_csv('./out.csv', header=None)
# print(df.columns)
# %%
# %%
x = []
y = []
for i, row in df.iterrows():
    x.append(row[0])
    y.append(row[1])

# %%
def find_min(x):
    min_x = 255
    for i in x:
        if i == -1:
            continue
        else:
            if i < min_x:
                min_x = i
    return min_x

# %%
def find_bounding_box():
    df = pd.read_csv('./out.csv', header=None)

    x = []
    y = []
    for i, row in df.iterrows():
        x.append(row[0])
        y.append(row[1])

    min_x = find_min(x)
    max_x = max(x)
    min_y = find_min(x)
    max_y = max(y)
    return min_x, max_x, min_y, max_y
