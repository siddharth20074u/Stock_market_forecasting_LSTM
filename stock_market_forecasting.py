#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:02:16 2020

@author: siddharthsmac
"""

! pip install pandas-datareader

import pandas_datareader as pdr

df = pdr.get_data_tiingo('AAPL', api_key = '')

df1 =df.reset_index()['close']

import matplotlib.pyplot as plt

plt.plot(df1)

## LSTM is sensitive to scale of the data

import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

## For Time-series, Sequence data

training_size = int(len(df1)*0.65)
test_size = len(df1) - training_size

train_data, test_data = df1[0 : training_size, :], df1[training_size : len(df1), :]

## Time-steps

import numpy

def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i : (i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)

time_step = 100

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

## reshape input data, which is required for LSTM

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (100, 1)))
model.add(LSTM(50, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.summary()

model.fit(X_train, y_train, validation_data = (X_test, y_test), 
          epochs  = 100, batch_size = 64, verbose = 1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

look_back = 100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2) + 1: len(df1) - 1, :] = test_predict

plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

X_input = test_data[len(test_data) - time_step :].reshape(-1, 1)

temp_input = list(X_input)

from numpy import array

lst_output = []
n_steps = 100
i = 0
while(i < 30):
    if (len(temp_input) > 100):
        X_input = np.array(temp_input[1:])
        print(' {} day input {}'.format(i, X_input))
        X_input = X_input.reshape(-1, 1)
        X_input = X_input.reshape(1, n_steps, 1)
        yhat = model.predict(X_input, verbose = 0)
        print('{} day output {}'.format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = 1 + 1
    else:
        X_input = X_input.reshape(1, n_steps, 1)
        yhat = model.predict(X_input, verbose = 0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        
day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

df3 = df1.tolist()
df3.extend(lst_output)

plt.plot(day_new, scaler.inverse_transform(df1[len(df1) - time_step :]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
