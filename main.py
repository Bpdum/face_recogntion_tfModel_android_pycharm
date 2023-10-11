import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use("seaborn-v0_8")
from tqdm import tqdm

# pathlib to my loccal file

files = os.listdir("/home/michael/Desktop/archive/dataset/")
files
# initialisation of array
image_array = []  # it's a list later i will convert it to array
label_array = []
path = "/home/michael/Desktop/archive/dataset/"
# loop through each sub-folder in train data
for i in range(len(files)):
    # files in sub-folder
    file_sub = os.listdir(path + files[i])

    for k in tqdm(range(len(file_sub))):
        try:
            img = cv2.imread(path + files[i] + "/" + file_sub[k])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96, 96))
            image_array.append(img)
            label_array.append(i)
        except:
            pass

# collecting garbae
gc.collect()
# importing process utils
import psutil

print(psutil.virtual_memory())
# image arrays
image_array = np.array(image_array) / 255.0
label_array = np.array(label_array)
# importing sckit data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(image_array, label_array, test_size=0.15)
from keras import layers, callbacks, utils, applications, optimizers
from keras.models import Sequential, Model, load_model

# checking file length
len(files)
model = Sequential()
# I will use MobileNetV2 as an pretrained model
pretrained_model = tf.keras.applications.EfficientNetB0(input_shape=(96, 96, 3), include_top=False,
                                                        weights="imagenet")
model.add(pretrained_model)
model.add(layers.GlobalAveragePooling2D())
# add dropout to increase accuracy by not overfitting
model.add(layers.Dropout(0.3))
# add dense layer as final output
model.add(layers.Dense(1))
model.summary()
# model seuen

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
# creating a chechpoint to save model at best accuarcy

ckp_path = "trained_model/model"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                      monitor="val_mae",
                                                      mode="auto",
                                                      save_best_only=True,
                                                      save_weights_only=True)

# create a lr reducer which decrease learning rate when accuarcy does not increase
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.9, monitor="val_mae",
                                                 mode="auto", cooldown=0,
                                                 patience=5, verbose=1, min_lr=1e-6)
# patience : wait till 5 epoch
# verbose : show accuracy every 1 epoch
# min_lr=minimum learning rate

# model train_test_split
EPOCHS = 300
BATCH_SIZE = 64

history = model.fit(X_train,
                    Y_train,
                    validation_data=(X_test, Y_test),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint, reduce_lr]
                    )

model.load_weights(ckp_path)

# converting
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

prediction_val = model.predict(X_test, batch_size=BATCH_SIZE)

# p@rediction
prediction_val[:20]

Y_test[:20]


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('hey am coding using python')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
