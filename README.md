# TSA_EX7 AUTO REGRESSIVE MODEL

## Date: 25-04-2025

## AIM:
To Implementat an Auto Regressive Model using Python

## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.

## PROGRAM:

```

Import necessary libraries :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

Read the CSV file into a DataFrame :

data = pd.read_csv('/content/AirPassengers.csv',parse_dates=['Month'],index_col='Month')


Perform Augmented Dickey-Fuller test :

result = adfuller(data['#Passengers']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])


Split the data into training and testing sets :

x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]


Fit an AutoRegressive (AR) model with 13 lags :

lag_order = 13
model = AutoReg(train_data['#Passengers'], lags=lag_order)
model_fit = model.fit()


Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :

plt.figure(figsize=(10, 6))
plot_acf(data['#Passengers'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plt.figure(figsize=(10, 6))
plot_pacf(data['#Passengers'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


Make predictions using the AR model :

predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)


Compare the predictions with the test data :

mse = mean_squared_error(test_data['#Passengers'], predictions)
print('Mean Squared Error (MSE):', mse)


Plot the test data and predictions :

plt.figure(figsize=(12, 6))
plt.plot(test_data['#Passengers'], label='Test Data - Number of passengers')
plt.plot(predictions, label='Predictions - Number of passengers',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Number of passengers')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()

```

## OUTPUT:

Dataset:

![image](https://github.com/user-attachments/assets/7d83ab7b-70c9-4efa-8f01-ed1541cb39a1)

ADF test result:

![image](https://github.com/user-attachments/assets/0130fe4f-eced-4e4d-a499-cca7aaf837df)

PACF plot:

![image](https://github.com/user-attachments/assets/24b58e49-3292-414c-a020-3c6239606fc7)

ACF plot:

![image](https://github.com/user-attachments/assets/fdd0968f-7ff5-4024-a3d2-a7d6f133f446)

Accuracy:

![image](https://github.com/user-attachments/assets/f67d8395-b446-4c2d-8ab5-d8439ada275f)

Prediction vs test data:

![image](https://github.com/user-attachments/assets/531a9aa2-65b5-458d-a8ac-bfb701a6390d)



## RESULT:
Thus we have successfully implemented the auto regression function using python.
