""" Dataset: https://www.kaggle.com/sumanthvrao/data"""

import matplotlib.pyplot as plt
import csv0
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from acc_plotter import plot_accuracy
from tensorflow.keras.optimizers import SGD


def plot_series(time, series, label=None, start=0, stop=None):
    plt.plot(time[start:stop], series[start:stop], label= label)
    plt.xlabel('Time')
    plt.ylabel('Average Temp')

train_path = r'daily_temp_dataset\DailyDelhiClimateTrain.csv'
test_path = r'daily_temp_dataset\DailyDelhiClimateTest.csv'

def get_data(data_path):
    time = 0
    series = []
    with open(train_path, 'r') as delhi_temp:
        reader = csv.reader(delhi_temp)
        next(reader)
        for row in reader:
            series.append(float(row[1]))
            time += 1
    with open(test_path, 'r') as delhi_temp2:
        reader = csv.reader(delhi_temp2)
        next(reader)
        for row in reader:
            series.append(float(row[1]))
            time += 1

    series = np.array(series)
    time = np.array(range(time))
    return time , series


time, series = get_data(train_path)
plot_series(time, series, label='Training Data')
plt.figure()
#plt.show()

split = 1000
train_time = time[:split]
test_time = time[split:]
train_series = series[:split]
test_series = series[split:]
test_series = np.array(test_series)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map( lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle_buffer).prefetch(1)

    return dataset

def prediction_data(model, series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)

    forecast = model.predict(dataset)
    return forecast

window_size = 32
batch_size = 1024
shuffle_buffer = 10000

train_dataset = windowed_dataset(train_series, window_size, batch_size, shuffle_buffer)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv1D(64, 5, padding='causal', activation='relu', input_shape=[None,1]), #64
        tf.keras.layers.Bidirectional(LSTM(120, return_sequences=True)), #64
        tf.keras.layers.Bidirectional(LSTM(64)), #32
        tf.keras.layers.Dense(64, activation='relu'), #32
        tf.keras.layers.Dense(32, activation='relu'), #12
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda( lambda x : x * 40.0)
    ]
)

#model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])
model.compile(optimizer= SGD(lr=4.7e-5), loss= tf.keras.losses.Huber(), metrics=['mae']) #4.7e-5
model.summary()

# todo optimize the Learning Rate
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))

history = model.fit(train_dataset, epochs=500, verbose=2) #, callbacks=[lr_scheduler])
plot_accuracy(history)
plt.figure()

# # todo plot lrs and loss
# lr = history.history['lr']
# loss = history.history['loss']
# plt.semilogx(lr, loss, label= 'LRs vs LOSS')
# plt.axis([1e-8, 1e-4, 0, 60])
# plt.show()

forecast = prediction_data(model, series[..., np.newaxis], window_size, batch_size)
forecast = forecast[split-window_size:,-1]

plot_series(test_time, test_series)
plot_series(test_time, forecast[1:])

mae = tf.keras.losses.mse(test_series, forecast[1:])
mse = tf.keras.losses.mse(test_series, forecast[1:])

print("MSE error : {}".format(mse))
print("MAE error : {}".format(mae))
plt.show()
