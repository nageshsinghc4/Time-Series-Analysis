#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:31:26 2019

@author: nageshsinghchauhan
"""

#The program forecast the consumption of electricity in coming future
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Above is a special style template for matplotlib, highly useful for visualizing time series data
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6

df = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/time_series/electricConsumption/Electric_Production.csv')


df.columns=['Date', 'Consumption']
df=df.dropna()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True) #set date as index
df.head()

#Visualize the Value
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.title("production graph")
plt.plot(df)


"""
Mean is not constant in this case as we can clearly see an upward trend. 
Hence, we have identified that our series is not stationary. 
We need to have a stationary series to do time series forecasting. 
In the next stage, we will try to convert this into a stationary series.
"""

#scatter plot of the Consumption
df.plot(style='k.')
plt.show()

#Distribution of the dataset
df.plot(kind='kde')
#We can observe a near-normal distribution(bell-curve) over sales values.


#To separate the trend and the seasonality from a time series, 
# we can decompose the series using the following code.


"""
The above code has a separated trend and seasonality for us.
This gives us more insight into our data and real-world actions.
Clearly, there is an upward trend and a recurring event where consumption shoot maximum every year!
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
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    
    #perform dickey fuller test  
    print("Results of dickey fuller test")
    adft = adfuller(timeseries['Consumption'],autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
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
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.plot(df_log)
plt.plot(moving_avg, color="red")
plt.plot(std_dev, color ="black")
plt.show()

"""
After finding the mean, we take the difference of the series and the mean at every point in the series.
This way, we eliminate trend out of a series and obtain a more stationary series.
"""
df_log_moving_avg_diff = df_log-moving_avg
df_log_moving_avg_diff.dropna(inplace=True)

"""
Perform dickey fuller test (ADFT) once again. 
This is the actual code for dickey fuller test. 
We have to perform this function everytime to check whether the data 
is stationary or not.
"""
test_stationarity(df_log_moving_avg_diff)
#From the above graph we observed that the data attained stationartiy. We also see that the test statistics and critial value is relatively equal

"""
We need to check the weighted average, to understand the trend of the data in timeseries. 
Take the previous log data nd perform the following operation.
"""
weighted_average = df_log.ewm(halflife=12, min_periods=0,adjust=True).mean()
print(weighted_average.head())

"""
The exponential moving average (EMA) is a weighted average of the last n prices, 
where the weighting decreases exponentially with each previous price/period. 
In other words, the formula gives recent prices more weight than past prices.
"""
plt.plot(df_log)
plt.plot(weighted_average, color='red')
plt.xlabel("Date")
plt.ylabel("Consumption")
from pylab import rcParams
rcParams['figure.figsize'] = 10,6
#plt.legend()
plt.show(block =False)

#Previously we subtracted data_logscale with moving average, now take the same log_scale and subtract with weighted_average
logScale_weightedMean = df_log-weighted_average
# use the same function defined above and pass the object into it.
from pylab import rcParams
rcParams['figure.figsize'] = 10,6
test_stationarity(logScale_weightedMean)




    



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
plt.title("Shifted timeseries")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.plot(df_log_diff)

#Let us test the stationarity of our resultant series
df_log_diff.dropna(inplace=True)

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
result = seasonal_decompose(df_log, model='additive', freq = 12)
result.plot()
plt.show()

trend = result.trend
trend.dropna(inplace=True)

seasonality = result.seasonal
seasonality.dropna(inplace=True)

residual = result.resid
residual.dropna(inplace=True)

test_stationarity(residual)


"""
After the decomposition, if we look at the residual then we have clearly a flat line for both 
mean and standard deviation. We have got our stationary series and now we can move to model building!!!
"""

#Forecasting 
"""
Before we go on to build our forecasting model, 
we need to determine optimal parameters for our model. For those optimal parameters, we need ACF and PACF plots.

A nonseasonal ARIMA model is classified as an “ARIMA(p,d,q)” model, where:

p is the number of autoregressive terms,
d is the number of nonseasonal differences needed for stationarity, and
q is the number of lagged forecast errors in the prediction equation.
Values of p and q come through ACF and PACF plots. So let us understand both ACF and PACF!
"""
"""
#Below code plots, both ACF and PACF plots for us
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
autocorrelation_plot(df_log)
plot_pacf(df_log, lags=10)
plt.show()
"""

# plot acf  and pacf graphs ( auto corellation function and partially auto corellation function )
# to find 'p' from p,d,q we need to use, PACF graphs and for 'q' use ACF graph
from statsmodels.tsa.stattools import acf,pacf
# we use d value here(data_log_shift)
acf = acf(df_log_diff, nlags=15)
pacf= pacf(df_log_diff, nlags=15,method='ols')

# ols stands for ordinary least squares used to minimise the errors

# 121 and 122 makes the data to look side by size 

#plot PACF
plt.subplot(121)
plt.plot(acf) 
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.title('Auto corellation function')
plt.tight_layout()


#plot ACF
plt.subplot(122)
plt.plot(pacf) 
plt.axhline(y=0,linestyle='-',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='black')
plt.title('Partially auto corellation function')
plt.tight_layout()


"""
What suggests AR(q) terms in a model?

ACF shows a decay
PACF cuts off quickly
What suggests MA(p) terms in a model?

ACF cuts off sharply
PACF decays gradually
In PACF, the plot crosses the first dashed line(95% confidence interval line) around lag 2 hence p=2

Below code fits an ARIMA model for us
"""

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_log, order=(3,1,3))
result_AR = model.fit(disp = 0)
plt.plot(df_log_diff)
plt.plot(result_AR.fittedvalues, color='red')
plt.title("sum of squares of residuals")
print('RSS : %f' %sum((result_AR.fittedvalues-df_log_diff["Consumption"])**2))
#RSS : 0.5227

# less the RSS more effective the model is


"""
forecast electricity consumption for next 4 months
"""
"""
future=df_log
future=future.reset_index()
mon=future["Date"]
mon=mon+pd.DateOffset(months=7)
future_dates = mon[-7-1:]
future = future.set_index('Date')
newDf = pd.DataFrame(index=future_dates, columns=future.columns)
future = pd.concat([future,newDf])
future["Forecast Consumption"]= result_AR.predict(start=35, end =43, dynamic=True)
future["Forecast Consumption"].iloc[-10:]=result_AR.forecast(steps=10)[0]
future[['Consumption','Forecast Consumption']].plot()
"""

# we founded the predicted values in the above code and we need to print the values in the form of series
ARIMA_predicts = pd.Series(result_AR.fittedvalues,copy=True)
ARIMA_predicts.head()

# finding the cummulative sum
ARIMA_predicts_cumsum = ARIMA_predicts.cumsum()
print(ARIMA_predicts_cumsum.head())


ARIMA_predicts_log = pd.Series(df_log['Consumption'],index = df_log.index)
ARIMA_predicts_log = ARIMA_predicts_log.add(ARIMA_predicts_cumsum,fill_value=0)
print(ARIMA_predicts_log.head())

# converting back to the exponential form results in getting back to the original data.
ARIMA_final_preditcs = np.exp(ARIMA_predicts_log)
rcParams['figure.figsize']=10,10
plt.plot(df)
plt.plot(ARIMA_predicts_cumsum)

plt.plot(ARIMA_predicts_cumsum)
plt.plot(df)

#future prediction
result_AR.plot_predict(1,500)
x=result_AR.forecast(steps=200)

# from the above graph, we calculated the future predictions till 2024
# the greyed out area is the confidence interval wthe predictions will not cross that area.

#Finally we calculated the units(value) of electricity is consumed in the coming future using time series analysis.






    


















