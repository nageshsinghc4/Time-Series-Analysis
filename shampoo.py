#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:25:55 2019

@author: nageshsinghchauhan
"""
#The program forecast the sales of Shampoo from a retail shop
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
from plotly import tools
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import math
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/time_series/shampoo/sales_shampoo.csv')

df.columns=['Month', 'Sales']
df=df.dropna()
df['Date'] = pd.to_datetime('200'+df.Month, format='%Y-%m')
df = df.drop(['Month'], axis=1)
df.set_index('Date', inplace=True) #set date as index
df.head()

#To visualise a time series, we can call the plot function directly
df.plot()

"""
Mean is not constant in this case as we can clearly see an upward trend. Hence, we have identified that our series is not stationary. We need to have a stationary series to do time series forecasting. In the next stage, we will try to convert this into a stationary series.
"""

df.plot(style='k.')
plt.show()

#We can observe a near-normal distribution(bell-curve) over sales values.
df.plot(kind='kde')

#To separate the trend and the seasonality from a time series, we can decompose the series using the following code.


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df, model='multiplicative')
result.plot()
plt.show()

"""
The above code has a separated trend and seasonality for us.
This gives us more insight into our data and real-world actions.
Clearly, there is an upward trend and a recurring event where shampoo sales shoot maximum every year!
"""

"""
we need to check if a series is stationary or not.
"""

"""
Following function is a one which can plot a series with it’s rolling mean and standard deviation.
 If both mean and standard deviation are flat lines(constant mean and constant variance),
the series become stationary!
"""
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean &amp; Standard Deviation')
    plt.show(block=False)
    
test_stationarity(df)
#we can see the increasing mean and standard deviation and hence our series is not stationary.

""" Eliminating trend """
"""
we start by taking a log of the series to reduce the magnitude of the values and 
reduce the rising trend in the series. Then after getting the log of the series, 
we find the rolling average of the series. A rolling average is calculated by taking 
input for the past 6 months and giving a mean sales value at every point further ahead in series.
"""
df_log = np.log(df)
moving_avg = df_log.rolling(6).mean()
plt.plot(df_log)
plt.plot(moving_avg, color="red")
plt.show()

"""
After finding the mean, we take the difference of the series and the mean at every point in the series.
This way, we eliminate trend out of a series and obtain a more stationary series.
"""
df_log_moving_avg_diff = df_log-moving_avg
df_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(df_log_moving_avg_diff)

"""
There can be cases when there is a high seasonality in the data. 
In those cases, just removing the trend will not help much. 
We need to also take care of the seasonality in the series. 
One such method for this task is differencing!
"""
"""
Differencing is a method of transforming a time series dataset. 
It can be used to remove the series dependence on time, so-called temporal dependence. 
This includes structures like trends and seasonality. Differencing can help stabilize the mean
of the time series by removing changes in the level of a time series, and so eliminating (or reducing) 
trend and seasonality.
Differencing is performed by subtracting the previous observation from the current observation.
"""
df_log_diff = df_log - df_log.shift()
plt.plot(df_log_diff)

#Let us test the stationarity of our resultant series
test_stationarity(df_log_diff)

#Now the series is stationary as both maean and std are constant

"""
Decomposition
It provides a structured way of thinking about a time series forecasting problem,
 both generally in terms of modelling complexity and specifically in terms of how to best capture
 each of these components in a given model.

"""
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_log, model='multiplicative')
result.plot()
plt.show()

trend = result.trend
seasonality = result.seasonal
residual = result.resid
test_stationarity(residual)

"""
After the decomposition, if we look at the residual then we have clearly a flat line for both 
mean and standard deviation. We have got our stationary series and now we can move to model building!!!
"""
#Forecasting 
"""
Before we go on to build our forecasting model, 
we need to determine optimal parameters for our model. For those optimal parameters, we need ACF and PACF plots.
"""
"""
A nonseasonal ARIMA model is classified as an “ARIMA(p,d,q)” model, where:

p is the number of autoregressive terms,
d is the number of nonseasonal differences needed for stationarity, and
q is the number of lagged forecast errors in the prediction equation.
Values of p and q come through ACF and PACF plots. So let us understand both ACF and PACF!
"""

#Below code plots, both ACF and PACF plots for us
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
autocorrelation_plot(df_log)
plot_pacf(df_log, lags=10)
plt.show()

"""
What suggests AR(q) terms in a model?

ACF shows a decay
PACF cuts off quickly
What suggests MA(p) terms in a model?

ACF cuts off sharply
PACF decays gradually
In PACF, the plot crosses the first dashed line(95% confidence interval line) around lag 4 hence p=4

Below code fits an ARIMA model for us
"""

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_log, order=(4,1,2))
result_AR = model.fit(disp=-1)
plt.plot(df_log_diff)
plt.plot(result_AR.fittedvalues, color='red')
plt.show()



"""
forecast shampoo sales for the next 4 months
"""

future=df_log
future=future.reset_index()
mon=future["Date"]
mon=mon+pd.DateOffset(months=7)
future_dates = mon[-7-1:]
future = future.set_index('Date')
newDf = pd.DataFrame(index=future_dates, columns=future.columns)
future = pd.concat([future,newDf])
future["Forecast Sales"]= result_AR.predict(start=35, end =43, dynamic=True)
future["Forecast Sales"].iloc[-10:]=result_AR.forecast(steps=10)[0]
future[['Sales','Forecast Sales']].plot()



















