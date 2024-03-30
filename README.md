# Stock Price Prediction

## AIM
To develop a Recurrent Neural Network model for stock price prediction.
## Problem Statement and Dataset
We aim to build a RNN model to predict the stock prices of Google using the dataset provided. The dataset has many features, but we will be predicting the "Open" feauture alone. We will be using a sequence of 60 readings to predict the 61st reading.
Note: These parameters can be changed as per requirements.
## NEURAL NETWORK MODEL
![image](https://github.com/Revathi-Dayalan/rnn-stock-price-prediction/assets/96000574/f1d3d776-0c6e-41f4-9cc1-c739626c0aed)
60 Inputs with 60 Neurons in the RNN Layer (hidden) and one neuron for the Output Layer.
![image](https://github.com/Revathi-Dayalan/rnn-stock-price-prediction/assets/96000574/f72727c4-b882-47f5-ad84-1cdf727c7549)
## Design Steps

### Step 1:
Read the csv file and create the Data frame using pandas.

### Step 2:
Select the " Open " column for prediction. Or select any column of your interest and scale the values using MinMaxScaler.

### Step 3:
Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train.

### Step 4:
Create a model with the desired number of nuerons and one output neuron.

### Step 5:
Follow the same steps to create the Test data. But make sure you combine the training data with the test data.

### Step 6:
Make Predictions and plot the graph with the Actual and Predicted values.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

df_train = pd.read_csv('trainset.csv')
df_train.head(60)

train_set = df_train.iloc[:,1:2].values
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)


X_train

X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train1

model = Sequential([layers.SimpleRNN(50,input_shape=(60,1)),
                    layers.Dense(1)
                    ])

model.compile(optimizer='Adam', loss='mae')

model.fit(X_train1,y_train,epochs=100,batch_size=32)

df_test=pd.read_csv("testset.csv")
test_set = df_test.iloc[:,1:2].values

dataset_total = pd.concat((df_train['Open'],df_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```
## Output
### True Stock Price, Predicted Stock Price vs time
![image](https://github.com/Revathi-Dayalan/rnn-stock-price-prediction/assets/96000574/25377ae6-03b5-4292-9af1-4bd4c7098fb8)
### Mean Square Error
![image](https://github.com/Revathi-Dayalan/rnn-stock-price-prediction/assets/96000574/083dc4bb-ba0c-4b93-9940-52d18cdaa78d)
## Result
A Recurrent Neural Network model for stock price prediction is developed.


