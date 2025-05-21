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
from statsmodels.tsa.statespace.sarimax import SARIMAX 
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, root_mean_squared_error


def ADF_test(df):
    p = adfuller(df)[1]
    print(f'P-value: {p:.4f} < 0.05')
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

def Model(train, test, order, exog_train=None, exog_test=None):
    if exog_train is not None and exog_test is not None:
        model = SARIMAX(train, exog=exog_train, order=order).fit()
        pred = model.predict(start=len(train), end=len(train) + len(test) - 1, exog=exog_test, typ='levels')
        model_type = 'ARIMAX'
        
        confidence = model.get_forecast(steps=len(test), exog=exog_test).conf_int()
        conf_int = confidence.values


        # Plotting for ARIMAX
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Training Data', color='blue')
        plt.plot(test.index, test, label='Actual Test Data', color='green')
        plt.plot(test.index, pred, label='Forecasted Test Data', color='red', linestyle='--')
        plt.axvline(x=train.index[-1], color='black', linestyle=':', label='Train-Test Split')
        plt.title(f'{model_type} Forecast vs Actual')
        plt.fill_between(pred.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        model = ARIMA(train, order=order).fit()
        pred = model.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels', dynamic=False)
        model_type = 'ARIMA'

        # Plotting for ARIMA
        pred.plot(legend=True, label='Forecasted Test Data', style='--')
        test.plot(legend=True, label='Actual Test Data')
        plt.title(f'{model_type} Forecast vs Actual')
        plt.grid(True)
        plt.show()

    pred = pred.rename('Forecast')
    return model, pred

def Evaluate(train, test, order, exog_train=None, exog_test=None):
    if exog_train is not None and exog_test is not None:
        model = SARIMAX(train, exog=exog_train, order=order).fit()
        pred = model.predict(start=len(train), end=len(train) + len(test) - 1, exog=exog_test, typ='levels')
        model_name = 'ARIMAX'
    else:
        model = ARIMA(train, order=order).fit()
        pred = model.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels', dynamic=False)
        model_name = 'ARIMA'

    pred = pred.rename('Forecast')

    mse = ((pred - test) ** 2).mean()
    rmse = mse ** 0.5
    mae = (abs(pred - test)).mean()
    mape = (abs(pred - test) / abs(test)).mean() * 100

    print(f'Evaluation of {model_name}{order}')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'AIC: {model.aic:.2f}')
    print(f'BIC: {model.bic:.2f}')

def Forecast(df, order, months, exog=None, future_exog=None):
    if exog is not None and future_exog is not None:
        final_model = SARIMAX(df, exog=exog, order=order).fit()
        forecast = final_model.forecast(steps=months, exog=future_exog)
    else:
        final_model = ARIMA(df, order=order).fit()
        forecast = final_model.forecast(steps=months)
    
    forecast = forecast.rename('Forecast')
    return forecast, final_model

def Forecasted_plot(actual, forecasted, conf_int=None):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', color='blue')
    plt.plot(forecasted, label='Forecasted', color='red')

    if conf_int is not None:
        plt.axvline(x=actual.index[-1], color='gray')
        plt.fill_between(forecasted.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')

        plt.gca().xaxis.set_major_locator(mdates.YearLocator())  
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator()) 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  

        plt.grid(True, which='minor', axis='x', linestyle='-', alpha=0.3) 
        plt.grid(True, which='major', axis='x', linestyle='-', alpha=0.7)
        plt.xticks(rotation=45)

    plt.title('Forecasted Price vs Actual Price')
    plt.ylabel('Price')
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

def Eval_possible_models(y_train, y_test, p_list, d_list, q_list, exog_train=None, exog_test=None, top_n=3):
    results = []

    for p in p_list:
        for d in d_list:
            for q in q_list:
                order = (p, d, q)
                try:
                    if exog_train is not None and exog_test is not None:
                        model = SARIMAX(y_train, exog=exog_train, order=order).fit()
                        pred = model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=exog_test, typ='levels')
                        model_name = 'ARIMAX'
                    else:
                        model = ARIMA(y_train, order=order).fit()
                        pred = model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, typ='levels', dynamic=False)
                        model_name = 'ARIMA'

                    pred = pred.rename('Forecast')

                    mse = ((pred - y_test) ** 2).mean()
                    rmse = mse ** 0.5
                    mae = (abs(pred - y_test)).mean()
                    mape = (abs(pred - y_test) / abs(y_test)).mean() * 100

                    results.append({
                        'Model': model_name,
                        'Order': order,
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape,
                        'AIC': model.aic,
                        'BIC': model.bic
                    })

                except Exception as e:
                    print(f"Failed for {order}: {e}")

    # Convert to DataFrame for easy sorting
    results_df = pd.DataFrame(results)

    # Sort by MAPE ascending
    top_mape = results_df.sort_values('MAPE').head(top_n)
    top_aic = results_df.sort_values('AIC').head(top_n)
    top_bic = results_df.sort_values('BIC').head(top_n)
    all_models = results_df 

    return all_models, top_mape, top_aic, top_bic 

def Confidence_intervals(model, steps, exog=None):
    if exog is not None:
        confidence = model.get_forecast(steps=steps, exog=exog).conf_int()
    else:
        confidence = model.get_forecast(steps=steps).conf_int()
      
    return confidence

def HoltWintersModel(train, test, seasonal_periods = 12, trend='add', seasonal='add'):
    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecast = model_fit.forecast(len(test))
    return forecast

def Evaulate_HW(test, pred):
    mape = mean_absolute_percentage_error(test, pred) * 100
    mae = mean_absolute_error(test, pred)
    mse = mean_squared_error(test, pred)
    rmse = root_mean_squared_error(test, pred)
    
    print(f'MAPE: {mape:.2f}%')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')

def HoltWintersForecast(df,months, seasonal_periods = 12, trend='add', seasonal='add'):
    model = ExponentialSmoothing(df, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    forecast = model_fit.forecast(months)
    return forecast

def HWForecast_plot(actual, forecast):
    forecast.plot(label='Forecast', color='red')
    actual.plot(label='Actual', color='blue')
    plt.legend()