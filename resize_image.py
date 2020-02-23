# %%
import pandas as pd
import numpy as np
import ast
import json

# %%
# df = pd.read_csv('./out.csv', header=None)
# print(df.columns)
# sample = pd.read_csv('./airplane.csv')

# %%
# first_df = df['drawing'][0]
# first_sample = sample['drawing'][1]
# %%
# first_sample = ast.literal_eval(first_sample)
# print(type(first_sample))
# %%
# for i in first_sample:
#     # print(type(i))
#     # i = json.load(i)
#     for j in i:
#         print(len(j))
#         # print(min(j))
#         # print(max(j))
# first = first_sample[0]
# print(first[0])

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
def reshape(x, y):
    min_x = find_min(x)
    max_x = max(x)
    min_y = find_min(y)
    max_y = max(y)
    for i in range(len(x)):
        if x[i] == -1:
            continue
        x[i] = x[i] - min_x
    for i in range(len(y)):
        if y[i] == -1:
            continue
        y[i] = y[i] - min_y
    for i in range(len(x)):
        if x[i] == -1:
            continue
        x[i] = int(x[i] / (max_x-min_x) * 235)
        x[i] = x[i] + 10
    for i in range(len(y)):
        if y[i] == -1:
            continue
        y[i] = int(y[i] / (max_y-min_y) * 235)
        y[i] = y[i] + 10

# %%
# x = []
# y = []
# for i, row in df.iterrows():
#     x.append(row[0])
#     y.append(row[1])

# %%
# print(x)

# %%
# reshape(x, y)

# %%
# print('X:', x)
# print('Y:', y)

# %%
# new_df = pd.DataFrame()
# new_df['X'] = x
# new_df['Y'] = y

# %%
# new_df.to_csv('new_out.csv', header=False, index=False)

# %%
def resize_image():
    df = pd.read_csv('./out.csv', header=None)

    x = []
    y = []
    for i, row in df.iterrows():
        x.append(row[0])
        y.append(row[1])

    reshape(x, y)

    new_df = pd.DataFrame()
    new_df['X'] = x
    new_df['Y'] = y

    new_df.to_csv('new_out.csv', header=False, index=False)