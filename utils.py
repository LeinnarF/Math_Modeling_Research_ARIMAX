import warnings
warnings.filterwarnings("ignore", message="No frequency information was provided, so inferred frequency MS will be used")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX as ARIMAX 

def ADF_test(df):
    p = adfuller(df)[1]
    print(f'P-value: {p} < 0.05')
    if p < 0.05:
        print('The time series is stationary')
    else:
        print('The time series is not stationary')

def Differencing(df, lags):
    df_diff = df.diff(lags).dropna()
    df_diff.plot(), ADF_test(df_diff)

def train_test_split(df, test_size):
    total = len(df)
    train_size = int(total - test_size)

    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    return train, test

def ACF_PACF(df, lags):
    plot_acf(df, lags=lags);
    plot_pacf(df, lags=lags);

def Order_RMSE(train, test, p_range, d_range, q_range):
    best_score, best_order = float("inf"), None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                order = (p,d,q)
                try:
                    model = ARIMA(train, order=order)
                    model_fit = model.fit()
                    predicted = model_fit.forecast(steps=len(test))
                    rmse = ((predicted - test) ** 2).mean() ** 0.5
                    if rmse < best_score:
                        best_score, best_order = rmse, order
                except:
                    continue
    print(f'Best ARIMA{best_order} RMSE={best_score}')

def Order_AIC(train, p_range, d_range, q_range):
    best_score, best_order = float("inf"), None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                order = (p,d,q)
                try:
                    model = ARIMA(train, order=order)
                    model_fit = model.fit()
                    aic = model_fit.aic
                    if aic < best_score:
                        best_score, best_order = aic, order
                except:
                    continue
    print(f'Best ARIMA{best_order} AIC={best_score}')

def Order_BIC(train, p_range, d_range, q_range):
    best_score, best_order = float("inf"), None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                order = (p,d,q)
                try:
                    model = ARIMA(train, order=order)
                    model_fit = model.fit()
                    bic = model_fit.bic
                    if bic < best_score:
                        best_score, best_order = bic, order
                except:
                    continue
    print(f'Best ARIMA{best_order} BIC={best_score}')

def Model(train, test, order):
    model = ARIMA(train, order=order).fit()

    start = len(train)
    end = start + len(test) - 1

    pred= model.predict(start=start, end=end, typ='levels', dynamic=False).rename('Forecast')

    pred.plot(legend=True)
    test.plot(legend=True)

    # Compare the forecast with the test set
    pred.apply(lambda x: f"{x:.2f}")
    for i in range(len(pred)):
        print(f'Actual: {test[i]}, Predicted: {pred[i]}')

    return model

def ModelX(train, exog_train, order):
    model = ARIMAX(train, order=order, exog=exog_train, seasonal_order=None).fit()
    return model

    


def Evaluate(train, test, order):
    model = ARIMA(train, order=order).fit()
    pred = model.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels', dynamic=False).rename('Forecast')

    mse = ((pred - test) ** 2).mean()
    rmse = mse ** 0.5
    mae = (abs(pred - test)).mean()
    mape = (abs(pred - test) / abs(test)).mean() * 100

    print(f'Evaluation of ARIMA{order}')
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}')
    print(f'AIC: {model.aic}')
    print(f'BIC: {model.bic}')

def Forecast(df, order, months):

    final_model = ARIMA(df, order = order).fit()
    forecast = final_model.forecast(steps=months).rename('Forecast')
    # forecast.apply(lambda x: f"{x:.2f}")
    
    return forecast, final_model

def Forecasted_plot(actual, forecasted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(forecasted, label='Forecasted', color='red')
    plt.title('Forecast vs Actual')
    plt.legend()
    plt.show()

def Residuals(model):
    residual=model.resid
    residual.plot()
    plt.axhline(y=0, color='red', linestyle='--')
    return residual

def LjungBox(residual, lags):
    lb = ljungbox(residual, lags, return_df=True)
    return lb

def BoxCox(df):
    df_boxcox, lambda_ = boxcox(df)
    df_boxcox = pd.Series(df_boxcox, index=df.index)
    return df_boxcox, lambda_

