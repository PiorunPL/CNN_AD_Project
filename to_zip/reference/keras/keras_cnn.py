import os
from memory_profiler import profile
import time
import keras
import numpy as np
import psutil
import tensorflow as tf
from keras import  layers
from keras import backend as K

class MemoryAndTimePrint(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time_epoch = time.time()
        self.memory_usage_epoch = psutil.Process(os.getpid()).memory_info().rss
        return super().on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        self.end_time_epoch = time.time()
        self.memory_usage_epoch = psutil.Process(os.getpid()).memory_info().rss - self.memory_usage_epoch
        print("\n=====================================")
        print("EPOCH SUMMARY")
        print("=====================================")
        print("Time: ", self.end_time_epoch - self.start_time_epoch)
        print("Memory: ", self.memory_usage_epoch/1024/1024, "MiB")
        print("=====================================\n")
        return super().on_epoch_end(epoch, logs)
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.memory_usage = psutil.Process(os.getpid()).memory_info().rss

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.memory_usage = psutil.Process(os.getpid()).memory_info().rss - self.memory_usage
        print("=====================================")
        print("TRAINING SUMMARY")
        print("=====================================")
        print("Time: ", self.end_time - self.start_time)
        print("Memory: ", self.memory_usage/1024/1024, "MiB")
        print("=====================================")

@profile
def fun():
    num_classes = 10
    input_shape = (28, 28, 1)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(6, kernel_size=(3, 3), activation="relu"), #Bias is true by default
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(84, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 100
    epochs = 5

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[MemoryAndTimePrint()])
    score = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

fun()
