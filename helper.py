import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_processor import DataProcessor
from statsmodels.tsa.stattools import adfuller
def download_and_process(ticker_dict, data_file):
    api_key = 'e8d134d6486a3fc5f3c97e9d44b86860'
    freq = input("Enter data frequency (e.g., '1d' for daily): ")
    dp = DataProcessor(ticker_dict, fred_api_key=api_key, interval=freq)

    dp = DataProcessor(ticker_dict, fred_api_key=api_key, interval=freq)
    dp.download_yf_data()
    dp.merge_all()
    dp.export()

    # Save to file for future use
    dp.data.to_parquet(data_file)
    return dp.data

# test for stationarity
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value}')
    if result[1] <= 0.05:
        print("The time series is stationary")
    else:
        print("The time series is non-stationary")
    
    return result[1] <= 0.05  # Return True if stationary, False otherwise


def generate_signals_from_multistep(test_true, test_pred, horizon=5, index=None):
    """
    Generate trading signals from multi-step LSTM forecasts.
    
    Parameters
    ----------
    test_true : np.ndarray
        Array of true values, shape (n_samples, forecast_horizon).
    test_pred : np.ndarray
        Array of predicted values, same shape as test_true.
    horizon : int
        How many steps ahead to compute percentage change (default=5).
    index : list or pd.DatetimeIndex, optional
        Index to use for the output DataFrame.
        
    Returns
    -------
    signals_df : pd.DataFrame
        Columns: ['start_price', 'pred_future_price', 'pct_change', 'signal']
    """
    
    n_samples, forecast_horizon = test_pred.shape
    signals_data = []

    for i in range(n_samples):
        start_price = test_true[i, 0]   # actual at time t
        if horizon < forecast_horizon:
            pred_future_price = test_pred[i, horizon-1]  # prediction at t+horizon
        else:
            continue  # skip if horizon beyond forecast length

        pct_change = (pred_future_price - start_price) / start_price
        signal = 1 if pct_change > 0 else -1

        signals_data.append((start_price, pred_future_price, pct_change, signal))

    signals_df = pd.DataFrame(
        signals_data,
        columns=["start_price", "pred_future_price", "pct_change", "signal"],
        index=index[:len(signals_data)] if index is not None else None
    )
    
    return signals_df

def backtest_from_signals(signals_df, test_true, horizon=5, initial_capital=10000, capital_deployed = 0.2):
    """
    Backtest strategy using signals and actual test_true prices.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Must contain columns ['start_price','signal'] from generate_signals_from_multistep.
    test_true : np.ndarray
        Array of true prices, shape (n_samples, forecast_horizon).
    horizon : int
        Horizon used for signals (default=5).
    initial_capital : float
        Starting capital.

    Returns
    -------
    results : pd.DataFrame
        Trade log with realized returns, strategy returns, and capital.
    """
    trade_log = []
    capital = initial_capital

    n_samples, forecast_horizon = test_true.shape

    # Initialize a zero series for per-step capital changes
    capital_changes = np.zeros(n_samples)
    for i in range(len(signals_df) - horizon + 1):
        start_price = test_true[i, 0]
        if horizon >= forecast_horizon:
            continue  # skip if horizon too large

        # enter at t + 1
        # exit at t + horizon
        future_price = test_true[i, horizon-1]

        realized_return = (future_price - start_price) / start_price
        strategy_return = signals_df.iloc[i]['signal'] * realized_return
        capital_change = capital_deployed * strategy_return

        # Update cumulative capital
        capital *= (1 + capital_change)
        capital_changes[i + horizon - 1] += capital_change  # add return at exit day

        trade_log.append({
            "Start_Price": start_price,
            "Future_Price": future_price,
            "Signal": signals_df.iloc[i]['signal'],
            "Return": strategy_return,
            "Capital_Change": capital_change,
            "Equity_After_Trade": capital
        })

        # Compute equity curve
    equity_series = initial_capital * (1 + pd.Series(capital_changes)).cumprod()

    return pd.DataFrame(trade_log), equity_series

def generate_ma_signals(test_true, sma_short, sma_long):
    """
    Generate signals from moving average crossover.

    Parameters
    ----------
    test_true : pd.Series or np.ndarray
        True price series for the test period.
    sma_short : pd.Series or np.ndarray
        Short-term moving average (e.g., sma_5 or ema_5).
    sma_long : pd.Series or np.ndarray
        Long-term moving average (e.g., sma_10 or ema_10).

    Returns
    -------
    signals_df : pd.DataFrame
        Columns: 'start_price', 'signal'
        signal = +1 (long), -1 (short)
    """
    
    test_true = pd.Series(test_true.squeeze())
    sma_short = pd.Series(sma_short.squeeze())
    sma_long = pd.Series(sma_long.squeeze())
    
    # Generate signals: +1 long if short MA > long MA, else -1
    signal = np.where(sma_short > sma_long, 1, -1)
    
    signals_df = pd.DataFrame({
        'start_price': test_true.values,
        'signal': signal
    })
    
    return signals_df

def compute_equity_from_signals(signals_df, test_true, initial_capital=10000, horizon=1):
    """
    Compute equity curve given signals and actual prices.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Columns: ['start_price', 'signal']
    test_true : np.ndarray or pd.Series
        True price series.
    initial_capital : float
        Starting capital.
    horizon : int
        Holding period in steps (default=1).

    Returns
    -------
    equity_df : pd.DataFrame
        Columns: ['Start_Price', 'Signal', 'Realized_Return', 'Strategy_Return', 'Capital']
    """
    test_true = pd.Series(test_true.squeeze())
    capital = initial_capital
    trade_log = []

    for i in range(len(signals_df) - horizon):
        start_price = test_true.iloc[i]
        future_price = test_true.iloc[i + horizon]
        realized_return = (future_price - start_price) / start_price
        strategy_return = signals_df.iloc[i]['signal'] * realized_return
        capital *= (1 + strategy_return)
        trade_log.append({
            'Start_Price': start_price,
            'Signal': signals_df.iloc[i]['signal'],
            'Realized_Return': realized_return,
            'Strategy_Return': strategy_return,
            'Capital': capital
        })
    
    equity_df = pd.DataFrame(trade_log)
    return equity_df