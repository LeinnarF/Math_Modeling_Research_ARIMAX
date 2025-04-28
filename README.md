# Forecasting Price of Refined Sugar in the Philippines using ARIMAX Model

_Research for the course MAT305 - Mathematical Modeling_

Sugar is one of the most consumed commodities in the Philippines, almost every household and establishment has sugar. It is one of the key ingredients in many products such as pastries, beverages, etc., from small scale business up to a business. Hence, a change in the monetary value of sugar may affect those businesses and the economy. There are many factors that affect the prices of sugar, taking all those factors into consideration may result in more accurate forecasting of the prices but with the cost of computational complexity. The study is about forecasting the prices of sugar in the Philippines using a mathematical model and taking some factors into account.

## Time Series Analysis

Time Series Analysis is a statistical and computational technique used to analyze data points collected or recorded at successive points in time. The goal is to identify patterns, trends, and relationships within the data over time, often to make predictions or gain insights.

Key Concepts:

1. Time-Ordered Data: The data is sequential, with each observation corresponding to a specific time point (e.g., daily stock prices, monthly sales, or yearly rainfall).
2. Trends: Long-term upward or downward movements in the data.
3. Seasonality: Regular, repeating patterns or cycles (e.g., increased sales during holidays).
4. Stationarity: A property where statistical characteristics (mean, variance) of the data remain constant over time.
5. Noise: Random variations or fluctuations in the data.

In this research we will focus on ARIMAX model

### ARIMA

ARIMA model is a combination of AR and MA model with I for differencing.

**AR(p)** - The AutoRegressive component checks the previous data on how it affects the current data.

**MA(q)** - The Moving Average component, it checks the errors on the previous data and how it affects the current data.

**I(d)** - The Integrated component is responsible for differencing the time series in order to achieve stationarity.

The ARIMA model compose of 3 parameters, (p, d, q) which corresponds to each component of the ARIMA model.

### ARIMAX

The ARIMAX model is the extension of ARIMA model by taking exogeneous variables into account. The inclusion of exogeneous variables may result to a much better forecasting than traditional ARIMA model.

### Modeling Process

The first step is to ensure that the time series data are stationary. One way to check the stationarity is through Augmented Dickey-Fuller test. The ADF test checks the existence of the unit root, if it exist the time series is not stationary.

We divide the time series data to a training set a testing set, 80% for training and 20% for testing.

To find the p and q for the AR and MA component, we use the ACF and PACF to find the possible lags for p and q.

We utilize a grid search to find the best lag order of p and q given the results of ACF and PACF. by iterating over and over until it finds the model with best AIC, BIC, and a relatively low MAPE

Once the lag order for p and q is identified, we proceed to Forecasting the time series data (not the training set) with the optimal p,d,q order.

However, if we forecast it right away, the exogeneous variable doesn't have a forecasted value. Thus, making the the exogeneous variable uses for forecasting the future. To solve it, the exogeneous variable must also be forecasted in order to fill the gap. We can use ARIMA or Holt-Winter exponential smoothing for forecasting the values of the exogeneous variables.

Once the Exogeneous variables are forecasted, we can now use it for the dependent variable.
