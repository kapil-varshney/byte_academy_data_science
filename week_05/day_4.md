## Time Series

A time series is a set of observations of a single variable at multiple different points in time. Time series data is different in that these observations <i>are</i> dependent on another variable. For example, the stock price of Microsoft today <i>is</i> related to the stock price yesterday.

### Stationarity 

A process is said to be <b>stationary</b> if the distribution of the observed values does <i>not</i> depend on time. For a stationary process, what we want is the distribution of the observed variable to be independent of time, so the mean and variance of our observations should be constant over time.

If we take the trends out of data, we can make it stationary, which then allows us to properly run regressions against other variables. Otherwise we would risk results that conflate the time trend with the effect of the other variables.  We can make data stationary by taking differences of the observations. 

### Autoregressive Model

In an autoregressive model, the response variable is regressed against previous values from the same time series. We say that a process if AR(1) if only the previous observed value is used. A process that uses the previous p values is called AR(p). A classic example of an AR(1) process is a random walk. In a random walk, a "walker" has an equal chance of stepping left or stepping right.

### Moving Average Model

A moving average model is similar to an autoregressive model except that instead of being based on the previous observed values, the model describes a relationship between an observation and the previous error terms. We say that a process is MA(1) if only the previous observed value is used. A process that uses the previous p values is called MA(p).

### Analysis

The New York Independent System Operator (NYISO) operates competitive wholesale markets to manage the flow of electricity across New York. We will be using this data, along with weather forecasts, to create a model that predicts electricity prices.

We begin by importing the needed modules and load the data:

``` python
import pandas as pd
import numpy as np
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (10, 6)
day_ahead_market = pd.read_csv('./day_ahead_market_lbmp.csv')
real_time_market = pd.read_csv('./real_time_market_lbmp.csv')
weather_forecast = pd.read_csv('./weather_forecast.csv')
```

We want to have the times in actual time objects, but right now they're strings, so we have to convert string date column to a datetime type:
``` python
day_ahead_market['Time Stamp'] = pd.to_datetime(day_ahead_market['Time Stamp'], format='%m/%d/%Y %H:%M')
real_time_market['Time Stamp'] = pd.to_datetime(real_time_market['Time Stamp'], format='%m/%d/%Y %H:%M:%S')
```

Now we have to do the same for the weather data:

``` python
weather_forecast['Forecast Date'] = pd.to_datetime(weather_forecast['Forecast Date'], format='%m/%d/%Y')
weather_forecast['Vintage Date'] = pd.to_datetime(weather_forecast['Vintage Date'], format='%m/%d/%Y')
weather_forecast['Vintage'] = weather_forecast['Vintage'].astype('category')
```

Now we're going to re-index the data by name of region and timestamp. This helps us manipulate and access the data. 

``` python
dam_time_name = day_ahead_market.set_index(['Name', 'Time Stamp'])
rtm_time_name = real_time_market.set_index(['Name', 'Time Stamp'])
```

We're only looking at the data for NYC, so we then select just the data for that:

``` python
dam_nyc_lbmp = dam_time_name['LBMP ($/MWHr)']['N.Y.C.']
rtm_nyc_lbmp = rtm_time_name['LBMP ($/MWHr)']['N.Y.C.']
```

So then we plot this to see the emerging relationships on the data that's a day ahead:

``` python
plt.figure(figsize=(10,8))
dam_nyc_lbmp.plot(title='NYC Day Ahead LBMP 2015')
plt.show()
```

And next for the data in real time:

``` python
plt.figure(figsize=(10,8))
rtm_nyc_lbmp.plot(title='NYC Realtime LBMP 2015')
plt.show()
```

The timestamps on the realtime data and day ahead data don't actually line up. The realtime data has observations every 5 minutes while the day ahead data has observations every hour. So we need to fix this by aligning the two series of data.

``` python
aligned_dam, aligned_rtm = rtm_nyc_lbmp.align(dam_nyc_lbmp, join='inner')
```

Next, we remote duplicates:

``` python
no_dup_al_dam = aligned_dam[~aligned_dam.index.duplicated(keep='first')]
no_dup_al_rtm = aligned_rtm[~aligned_dam.index.duplicated(keep='first')]

no_dup_al_dam.name = 'dam_lbmp'
no_dup_al_rtm.name = 'rtm_lbmp'
```

Next step is to insert this data into a dataframe. The dataframe is too wide, however, so we transpose it:

``` python
dam_rtm_df = pd.DataFrame([no_dup_al_dam, no_dup_al_rtm]).transpose()
```

Now that we have our pricing data for the NYC region, we need to get the weather data ready. The weather data comes from a different data source, and unfortunately it's not split into the same exact regions as the pricing data. To remedy this, we'll pick two weather stations nearby - the ones located at JFK airport and LGA airport - and average the temperatures together.

This gets all the temperature data from LGA and JFK stations: 
``` python
lga_and_jfk_indexed = weather_forecast[(weather_forecast['Station ID'] == 'LGA') |
                                       (weather_forecast['Station ID'] == 'JFK')].set_index(['Forecast Date',
                                                                                             'Vintage Date',
                                                                                             'Vintage',
                                                                                             'Station ID'])
```

We still have to prep our data a bit more. So first we unindex, which will flatten our dataframe. Next, we pick out the rows with the Vintage being 'Actual' because we've got a few different kinds of dates floating around and it's important to stay consistent. Our end goal is to have our data like:

```
Time stamp: today
Temperature: the actual temperature today (x)
Day ahead price: today's price for electricity tomorrow (x)
Realtime price: tomorrrow's electricity price observed tomorrow (y)
```

To accomplish this, we write:

``` python
mean_nyc_indexed = lga_and_jfk_indexed.mean(level=[0,1,2])
mean_nyc = mean_nyc_indexed.reset_index()
actual_temp_df = mean_nyc[mean_nyc['Vintage'] == 'Actual'] \
    .groupby(['Vintage Date']).first() \
    .rename(columns=lambda x: 'Actual ' + x) # prepend the word Actual to column names
dam_rtm_act_df = dam_rtm_df.join(actual_temp_df, how='left').fillna(method='ffill').dropna()
```

Next, observe that there is a different day ahead price every single hour, but only a single temperature estimate per day. We'll do something similar to the frequency mismatch we saw with day ahead/realtime data - resample at the lower frequency and average the data in those lower frequency buckets.

``` python
daily_df = dam_rtm_act_df.resample('D', how='mean')
```

The next question is whether to use the max temp or min temp. Let's take a look at the data to see if one or the other makes a difference: 

``` python
plt.figure(figsize=(14,10))
plt.plot_date(daily_df.index, daily_df['rtm_lbmp'], '-', label='RTM LBMP')
plt.plot_date(daily_df.index, daily_df['dam_lbmp'], '-', label='DAM LBMP')
plt.plot_date(daily_df.index, daily_df['Actual Min Temp'], '-', label='Min Temp')
plt.plot_date(daily_df.index, daily_df['Actual Max Temp'], '-', label='Max Temp')
plt.legend()
plt.show()
```

Min and max temperature seem to be so highly correlated with each other that it most likely won't matter one way or another which we used.


#### Model Estimation

Now that our data is in the format we want, we can complete a regression called ARIMA. It's a combination of 3 types of models - AutoRegressive (a combination of the previous values), Integrated (differencing the data), Moving Average (smoothing the data). It's typically represented as:

```
arima(# of AR terms, # of differences, # of smoothing terms) = arima(p, d, q)
```

So how can we go about picking our parameters? A common approach is to use the Box-Jenkins Method for parameter selection. First, you pick parameter d. You can do this using a Dickey-Fuller test. Then, you can use autocorrelation function (ACF) and partial autocorrelation function (PACF) to identify the AR parameter (p) and the MA parameter (q).

For our example, we'll just pick ARIMA(0,0,1) for its simplicity (this is therefore just an MA(1) process).

Before fitting the model, we'll split our data into the training and test data:

``` python
exog_data = np.array([daily_df['Actual Max Temp'].values, daily_df['dam_lbmp'].values])
```

Now we can fit the model on a portion of the data
``` python
k = 250
m = ARIMA(daily_df['rtm_lbmp'].values[0:k], [0,0,1], exog=np.transpose(exog_data[:,0:k]), dates=daily_df.index.values[0:k])
results = m.fit(trend='nc', disp=True)
```

Let's look at the predicted prices:

``` python
predicted_prices = results.predict(10, 364, exog=np.transpose(exog_data), dynamic=True)
```

Now let's look at it graphically:
``` python
plt.figure(figsize=(14, 10))
plt.plot(predicted_prices, label='prediction')
plt.plot(daily_df['rtm_lbmp'].values, label='actual RTM')
plt.legend()
plt.show()
```

#### Analysis

Our model seems to work well from the graph of the forecast. Looking at the coefficients from the ARIMA model, we can see that increasing temperatures and increasing day ahead prices are both associated with higher realtime prices the next day. Just by looking at the coefficients, you can see that our forecast is extremely similar to the day ahead values, with a small adjustment based on temperature and another adjustment based on the moving average term.

``` python
plt.figure(figsize=(14, 10))
plt.plot(predicted_prices, label='prediction')
plt.plot(daily_df['rtm_lbmp'].values, label='actual RTM')
plt.plot(daily_df['dam_lbmp'].values, label='actual DAM')
plt.legend()
plt.show()
```
Our data and model helps us predict the realtime electricity cost the day before. If you ran a company with significant power demands, you might be able to use such a model to decide whether or not to buy electricity in advance or on demand. Let's see how it would work:

``` python
len(predicted_prices)
len(daily_df['rtm_lbmp'].values[10:])

print("--- Trading Log ---")

i = 251
PnL = np.zeros(100)


while i < 351:
    if (predicted_prices[i] < daily_df['dam_lbmp'].values[i]) and (daily_df['rtm_lbmp'].values[i+1] < daily_df['dam_lbmp'].values[i]):
        
        # if our model says the DAM is overpriced, then don't pre-buy and buy at the realtime price
        
        print("Buy RTM, +", daily_df['dam_lbmp'].values[i] - daily_df['rtm_lbmp'].values[i+1])
        PnL[i-251] = daily_df['dam_lbmp'].values[i] - daily_df['rtm_lbmp'].values[i+1]
        
    elif (predicted_prices[i] > daily_df['dam_lbmp'].values[i]) and (daily_df['rtm_lbmp'].values[i+1] > daily_df['dam_lbmp'].values[i]):
        
        # if our model says the DAM is underpriced, pre-buy the electricity so you don't have to pay realtime price 
        
        print("Buy DAM, +", daily_df['rtm_lbmp'].values[i+1] - daily_df['dam_lbmp'].values[i] )
        PnL[i-251] = daily_df['rtm_lbmp'].values[i+1] - daily_df['dam_lbmp'].values[i]
        
    else:
        
        # if we were wrong, we lose money :(
        
        print("Lose $$, -", max(daily_df['rtm_lbmp'].values[i+1] - daily_df['dam_lbmp'].values[i],daily_df['dam_lbmp'].values[i] - daily_df['rtm_lbmp'].values[i+1]))
        PnL[i-251] = min(daily_df['rtm_lbmp'].values[i+1] - daily_df['dam_lbmp'].values[i],daily_df['dam_lbmp'].values[i] - daily_df['rtm_lbmp'].values[i+1])
    i = i+1
```

Now lets take a look at the visualization:

``` python
cumPnL = np.cumsum(PnL)
plt.figure(figsize=(10, 8))
plt.plot(cumPnL, label='PnL')
plt.legend()
plt.show()
```

Looks like the model works because it predicts every so slightly closer to the true RTM than the DAM!

``` python
dam_adj = daily_df['rtm_lbmp'].values[10:]-daily_df['dam_lbmp'].values[:-10]
mod_adj = daily_df['rtm_lbmp'].values[10:]-predicted_prices[:-1]
plt.figure(figsize=(10, 8))
plt.plot(dam_adj, label='DAM error')
plt.plot(mod_adj, label='Model error')
plt.legend()
plt.show()
```


## Stepwise Regression

This form of regression is used when we deal with multiple independent variables. In this technique, the selection of independent variables is done with the help of an automatic process, which involves no human intervention.

Stepwise Regression is sensitive to initial inputs, which can be mitigated by repeatedly running the algorithm on bootstrap samples.

### Bootstrap

R has a package called `bootStepAIC()` that implements a Bootstrap procedure to investigate the variability of model selection with the function `stepAIC()`.

Using the `stepAIC()` function, you can input an already fitted lm/glm model and the associated dataset. We’ll use the BostonHousing dataset from the mlbench package to showcase this:

``` R
library(bootStepAIC)
library(plotly)
library(mlbench)
 
# Load Boston housing dataset
data("BostonHousing")
 
# Fit Linear regression model
fit <- lm(crim ~ ., data = BostonHousing)
 
# Run bootstrapped stepwise regression
fit.boot <- boot.stepAIC(fit, data = BostonHousing, B = 100)
```

This gives us the following information:

- No of times a covariate was featured in the final model from `stepAIC()`
- No of times a covariate’s coefficient sign was positive / negative
- No of times a covariate was statistically significant (default at alpha = 5%)



We do this by observing statistical values like R-square, t-stats, and AIC metric to discern significant variables. Stepwise regression basically fits the regression model by adding/dropping co-variates one at a time based on a specified criterion. Some of the most commonly used Stepwise regression methods are:

- Standard stepwise regression does two things: it adds and removes predictors as needed for each step.
- Forward selection starts with most significant predictor in the model and adds variable for each step.
- Backward elimination starts with all predictors in the model and removes the least significant variable for each step.

The aim of this modeling technique is to maximize the prediction power with minimum number of predictor variables. It is one of the methods to handle higher dimensionality of data set.
