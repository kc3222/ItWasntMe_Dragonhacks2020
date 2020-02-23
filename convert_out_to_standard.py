# %%
import pandas as pd
import numpy as np
import ast
import json

# %%
df = pd.read_csv('./new_out.csv', header=None)
# print(df.columns)
# sample = pd.read_csv('./airplane.csv')

# %%
all_x_y = []
x = []
y = []
for i, row in df.iterrows():
    if row[0] < 0:
        # print('Hello')
        all_x_y.append([x, y])
        x = []
        y = []
    else:
        x.append(row[0])
        y.append(row[1])

# %%
new_df = pd.DataFrame()
new_df['drawing'] = [all_x_y, all_x_y]
new_df.to_csv('out_standard.csv', index=False)

# %%
