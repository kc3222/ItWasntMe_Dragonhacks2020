# %%
import pandas as pd
import numpy as np
import ast
import cv2
from keras.applications.mobilenet import preprocess_input
import keras
import tensorflow as tf
import os

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

# %%
DP_DIR = '../input/small-simplified-quick-draw-doodle-challenge/train_simplified_divided'  # thư mục chứa dữ liệu

BASE_SIZE = 256 # kích thước gốc của ảnh
NCSVS = 100 # số lượng files csv mà chúng ta đã chia ở bước trên
NCATS = 340 # số lượng category (số lớp mà chúng ta cần phân loại)
STEPS = 1000 # số bước huấn luyện trong 1 epoch
EPOCHS = 30 # số epochs huấn luyện
size = 128 # kích thước ảnh training đầu vào
batchsize = 32
np.random.seed(seed=42) # cài đặt seed 
tf.set_random_seed(seed=42) # cài đặt seed 

# %%
import ast

# owls = pd.read_csv('../input/quickdraw-doodle-recognition/train_simplified/owl.csv')
# owls = pd.read_csv('./test.csv')
# owls = pd.read_csv('./out_standard.csv')
# print(owls.columns)
# owls = owls[owls.recognized]
# owls['timestamp'] = pd.to_datetime(owls.timestamp)
# owls = owls.sort_values(by='timestamp', ascending=False)[-100:]
# owls['drawing'] = owls['drawing'].apply(ast.literal_eval)

# owls.head()

# %%
import matplotlib.pyplot as plt

# n = 2
# fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))

# print(len(owls.drawing))
# for i, drawing in enumerate(owls.drawing):
#     ax = axs[i // n, i % n]
#     print("i", i)
#     print("i // n", i//n)
#     print("i % n", i%n)
#     for x, y in drawing:
#         ax.plot(x, -np.array(y), lw=3)
#     ax.axis('off')
# fig.savefig('owls.png', dpi=200)
# plt.show()

# %%
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

# %%
from keras.models import load_model

densenet_model = load_model('../cifar10-models/xception_accuracy_0.8062352942.hdf5', custom_objects={'top_3_accuracy': top_3_accuracy, 'top_k_categorical_accuracy': top_k_categorical_accuracy})

# %%
BASE_SIZE = 256 # kích thước gốc của ảnh
NCSVS = 100 # số lượng files csv mà chúng ta đã chia ở bước trên
NCATS = 340 # số lượng category (số lớp mà chúng ta cần phân loại)
STEPS = 1000 # số bước huấn luyện trong 1 epoch
EPOCHS = 30 # số epochs huấn luyện
size = 128 # kích thước ảnh training đầu vào
batchsize = 32
np.random.seed(seed=42) # cài đặt seed 
tf.set_random_seed(seed=42) # cài đặt seed 

# %%
def draw_cv2(raw_strokes, size=128, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                
                yield x, y

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        temp = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
        x[i, :, :, 0] = temp
        print('Hello')
        plt.figure()
        plt.imshow(temp)
        plt.show()
    x = preprocess_input(x).astype(np.float32)
    return x

# %%
# test_df = pd.read_csv(os.path.join("./out_standard.csv"))
# test_drawing = df_to_image_array_xd(test_df, 128)
# print(test_drawing.shape)

# %%
# print(test_drawing)

# %%
# predictions = densenet_model.predict(test_drawing)
# print(predictions.shape)

# %%
def f2cat(filename: str) -> str:
    return filename.split('.')[0]

class Simplified():
    def __init__(self, input_path='./input'):
        self.input_path = input_path

    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return sorted([f2cat(f) for f in files], key=str.lower)

    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(json.loads)
        return df

# %%
# Get categories
NCSVS = 100
# s = Simplified('../quick-draw-doodle')
# categories = s.list_all_categories()
# categories_df = pd.DataFrame()
# categories_df['categories'] = [categories]
# categories_df.to_csv('categories.csv', index=False)
categories_df = pd.read_csv('./categories.csv')
categories = ast.literal_eval(categories_df['categories'][0])
# print(type(categories))
# print(len(categories))
for i in range(len(categories)):
    categories[i] = categories[i].replace(' ', '_')

# %%
# for i in categories:
#     if i == 'house':
#         print(i)

# %%
# for i, prediction in enumerate(predictions):
#     top_3_predictions = prediction.argsort()[-3:][::-1]
#     # print(categories[top_3_predictions[0]] + ' ' + categories[top_3_predictions[1]] + ' ' + categories[top_3_predictions[2]])
#     print(categories[top_3_predictions[0]])
#     print(categories[top_3_predictions[1]])
#     print(categories[top_3_predictions[2]])

# %%
def predict_image():
    test_df = pd.read_csv(os.path.join("./out_standard.csv"))
    test_drawing = df_to_image_array_xd(test_df, 128)
    predictions = densenet_model.predict(test_drawing)

    for i, prediction in enumerate(predictions):
        top_3_predictions = prediction.argsort()[-3:][::-1]
    return [categories[top_3_predictions[0]], categories[top_3_predictions[1]], categories[top_3_predictions[2]]]

# print(predict_image())