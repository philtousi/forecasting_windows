import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error as mse
import pandas_datareader as web
import datetime as dt
import warnings
warnings.filterwarnings("ignore")


def forecast_roll_vs_exp(data: pd.DataFrame, window_size: int, num_oos: int = 120, h: int = 1):

    forecast_roll = []
    forecast_exp = []
    testset = data.iloc[-num_oos:].squeeze().to_list()

    for i in range(-num_oos, 0, h):

        # Fit the model
        model_roll = AR(data[i-window_size:i], lags=1).fit()
        model_exp = AR(data[:i], lags=1).fit()

        # Predict the next value
        pred_roll = model_roll.forecast(h)
        pred_exp = model_exp.forecast(h)

        forecast_roll.append(pred_roll)
        forecast_exp.append(pred_exp)

    # MSEs
    mse_roll = mse(testset, forecast_roll)
    mse_exp = mse(testset, forecast_exp)
    mse_relative = mse_exp / mse_roll

    return round(mse_relative, 3)


def main():

    START = dt.datetime(1960, 1, 1)
    END = dt.datetime(2022, 8, 1)

    cpi = web.DataReader(["CPIAUCSL"], "fred", START, END)

    # Ensure stationarity
    cpi = np.log(cpi).diff(1)
    cpi.dropna(inplace=True)
    cpi.freq = cpi.index.inferred_freq

    w = 60
    mse_relative = forecast_roll_vs_exp(cpi, w)
    print(f'Relative MSE: {mse_relative}, window_size: {w}')

    for w in range(120, cpi.shape[0]//2, 60):
        mse_relative = forecast_roll_vs_exp(cpi, w)
        print(f'Relative MSE: {mse_relative}, window_size: {w}')


if __name__ == '__main__':
    main()
