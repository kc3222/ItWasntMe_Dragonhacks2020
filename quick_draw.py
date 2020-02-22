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
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x

# %%
