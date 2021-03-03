#!/usr/bin/env python
import click
import pandas as pd
import requests
import io
import datetime
import pandas_datareader.data as pdr
import sys
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Masking
from keras.optimizers import Adam
from keras.layers import Flatten

#ml function
def lstm(data, BS=3, EPOCHS=20, lr = 0.001, decay = 0.23):
    print("LSTM running ... ...")
    scaler = MinMaxScaler(feature_range = (0, 1))
    n, m = data.shape
    #data["Pre-Close"] = 0
    #for i in range(1, n):
    #    data["Pre-Close"].iloc[i] = data["Close"].iloc[i-1].copy()
    #data = data.iloc[1:n].copy()
    n, m = data.shape
    train = data.iloc[0:(n*3)//4].to_numpy()
    val = data[(n*3)//4:n].to_numpy()
    train_scale = scaler.fit_transform(train)
    val_scale = scaler.transform(val)
    timesteps = 1 ## for future fine tuning                                                                                                                                                       
    features_set = np.delete(train_scale, 3, axis=1).copy() #.append(apple_training_scaled[i, 0:apple_train.shape[1]-1])
    labels = train_scale[:, 3].copy()
    features_set_val =  np.delete(val_scale, 3, axis=1).copy() #get the features for training
    labels_val =  val_scale[:, 3].copy()       #get the prediction result for training
    features_set = np.reshape(features_set, (features_set.shape[0], timesteps, features_set.shape[1]))
    features_set_val = np.reshape(features_set_val, (features_set_val.shape[0], timesteps, features_set_val.shape[1]))
    opt = Adam(lr=lr, decay=decay / (EPOCHS))
    model = Sequential()
    model.add(LSTM(units=80, return_sequences=True, input_shape=(features_set.shape[1],features_set.shape[2])))#set first layer and the feature shape for each row
    model.add(LSTM(units=50, return_sequences=True))#set first layer and the feature shape for each row
    model.add(LSTM(units=50, return_sequences=True))#set first layer and the feature shape for each row
    model.add(Dropout(0.29))
    model.add(Flatten())
    model.add(Dense(units = 1,activation='relu'))
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=["binary_crossentropy"])
    checkpoint_filepath = 'checkpoint.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    history = model.fit(features_set, labels, epochs = EPOCHS, batch_size = BS, validation_data=(features_set_val, labels_val), callbacks=[model_checkpoint_callback],verbose=0)
    predictions = model.predict(features_set_val)
    predictions = np.reshape(predictions, (features_set_val.shape[0]))
    validation = val_scale.copy()
    validation[:,3] = predictions
    result_val = scaler.inverse_transform(validation)
    print(result_val[-1, 3])
    return result_val[-1, 3]

@click.command()
@click.option("--name")
def hello(name="AAPL"):
    """Return a table of 'value' stock price features"""

    #stock name
    stock = name
    
    #set up the start and end time point I want
    end = datetime.date.today()
    start = end + datetime.timedelta(days=-365)
    
    #using data_reader to retrieve stock data from yahoo finance
    data = pdr.DataReader(stock, 'yahoo', start, end)
    
    #data preprocessing
    data = data.round(3)
    #make model based on the original data
    tomorrow = lstm(data)

    #change data type and index of the datafram for better visualization
    data["Volume"] = data.apply(lambda x: "{:,.0f}".format(x["Volume"]), axis=1)
    data = data.reset_index()
    
    #return the table into html format and modify the look
    #return_table = data.to_html(table_id=stock, justify="center")
    #return_table = return_table[:6] + " align = 'center'" + return_table[6:]
    
    if tomorrow > data["Close"].iloc[-1]:
        future = 'Next: Bull'
    elif tomorrow < data["Close"].iloc[-1]:
        future = 'Next: Bear'
    else: future = 'Next: Same'
    
    # add header
    title = '{0} historical stock price (from {1} to {2})'.format(stock, start, end)

    #subtitle = '<h2 align="center"> Python rt = {0}, tf = {1} </h2>'.format(sys.version, tf.__version__)
    
    #return title + subtitle + future + return_table#  + future
    click.echo(f'{title}\n{future}!')

if __name__ == '__main__':
    #pylint: disable=no-value-for-parameter
    hello()