# %%
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import datetime as dt
from urllib.parse import urlparse
import os

class DataProcessor:
    def __init__(self, ticker_dict, fred_api_key, start_date="2011-01-04", perdiod="180d", interval="1h", tareget_col='audusd'):
        self.ticker_dict = ticker_dict
        self.start_date = start_date
        self.fred = Fred(api_key=fred_api_key)
        self.data = None
        self.period = perdiod
        self.interval = interval
        self.target_col = tareget_col

    # ---------- Yahoo Finance ----------
    def download_yf_data(self):
        data = {}
        cols = []
        for ticker in self.ticker_dict:
            try:
                if self.interval == "1h":
                    df = yf.download(ticker, period=self.period, interval=self.interval)
                else:
                    df = yf.download(ticker, start=self.start_date)
                if not df.empty:
                    data[ticker] = df['Close']
                    # if containing na values, fill them 
                    print(f"✓ {ticker}: {len(df)} records")
                    cols.append(self.ticker_dict[ticker])
                else:
                    print(f"✗ {ticker}: No data")
            except Exception as e:
                print(f"✗ {ticker}: Error - {e}")

        combined = pd.concat(data, axis=1)
        combined['date'] = combined.index
        combined.columns = cols + ['date']
        combined = combined[['date'] + cols]
        combined.ffill(inplace=True)
        self.data = combined.reset_index(drop=True)
        return self.data

    # ---------- Load URL Data ----------
    @staticmethod
    def load_data_from_url(url):
        parsed_url = urlparse(url)
        ext = os.path.splitext(parsed_url.path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(url)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(url)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    # ---------- RBA interest rate ----------
    def load_rba_rate(self):
        end = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        rba_url = f"https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx?v={end}"
        ir_au = self.load_data_from_url(rba_url)
        ir_au = ir_au.rename(columns={
            "F1 INTEREST RATES AND YIELDS – MONEY MARKET": "date",
            "Unnamed: 1": "rates"
        }).iloc[10:][['date', 'rates']]
        ir_au['date'] = pd.to_datetime(ir_au['date'])
        ir_au['rates'] = pd.to_numeric(ir_au['rates'])
        ir_au = ir_au[ir_au['date'] >= pd.to_datetime(self.start_date)]
        ir_au.set_index('date', inplace=True)
        return ir_au
    # ---------- Fed rates ----------
    def load_fed_rate(self):
        fed = self.fred.get_series('FEDFUNDS', self.start_date, dt.datetime.now())
        df_fed = pd.DataFrame(fed).reset_index()
        df_fed.columns = ['date', 'fed_rates']
        df_fed['date'] = pd.to_datetime(df_fed['date'])
        df_fed['fed_rates'] = pd.to_numeric(df_fed['fed_rates'])
        # Resample to daily
        full_dates = pd.DataFrame({'date': pd.date_range(df_fed['date'].min(), df_fed['date'].max())})
        df_fed = full_dates.merge(df_fed, on='date', how='left')
        df_fed['fed_rates'] = df_fed['fed_rates'].ffill()
        df_fed.set_index('date', inplace=True)
        return df_fed

    # ---------- CPI ----------
    def load_cpi(self, series_id, col_name):
        cpi = self.fred.get_series(series_id, self.start_date, dt.datetime.now())
        df_cpi = pd.DataFrame(cpi).reset_index()
        df_cpi.columns = ['date', col_name]
        df_cpi['date'] = pd.to_datetime(df_cpi['date'])
        df_cpi[col_name] = pd.to_numeric(df_cpi[col_name])
        df_cpi.set_index('date', inplace=True)
        return df_cpi

    # ---------- Merge all ----------
    def merge_all(self):
        rba = self.load_rba_rate()
        fed = self.load_fed_rate()
        au_cpi = self.load_cpi('FPCPITOTLZGAUS', 'cpi')
        us_cpi = self.load_cpi('FPCPITOTLZGUSA', 'us_cpi')
        indicators = self.create_technical_indicators(self.data, self.target_col)
        df = self.data.join(indicators)
        df = df.ffill().bfill()
        df['date'] = df['date'].dt.tz_convert(None)  # Remove timezone info if any

        if self.interval == "1h":
            # Reindex daily data to hourly frequency and forward fill
            print("Resetting index:")
            df.set_index('date', inplace=True)
            rba = rba.reindex(df.index, method='ffill')
            fed = fed.reindex(df.index, method='ffill')
            au_cpi = au_cpi.reindex(df.index, method='ffill')
            us_cpi = us_cpi.reindex(df.index, method='ffill')

        df = df.merge(rba, on='date', how='left')
        df = df.merge(fed, on='date', how='left')
        df = df.merge(au_cpi, on='date', how='left')
        df = df.merge(us_cpi, on='date', how='left')

        # Fill missing values
        for col in ['rates', 'fed_rates', 'cpi', 'us_cpi']:
            df[col] = df[col].ffill()
            df[col] = df[col].fillna(df[col].mean())

        self.data = df
        return self.data

    # ---------- Export ----------
    def export(self, filename='data.parquet'):
        self.data.to_parquet(filename)
        print(f"Data exported to {filename}")

    def create_technical_indicators(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Create a small set of technical indicators for a univariate target.
        """
        indicators = pd.DataFrame(index=df.index)

        # Moving Averages
        indicators['sma_5'] = df[target].rolling(window=5, min_periods=5).mean()
        indicators['sma_10'] = df[target].rolling(window=10, min_periods=10).mean()
        indicators['ema_5'] = df[target].ewm(span=5, adjust=False).mean()
        indicators['ema_10'] = df[target].ewm(span=10, adjust=False).mean()

        # Momentum
        indicators['momentum_5'] = df[target] - df[target].shift(5)
        indicators['momentum_10'] = df[target] - df[target].shift(10)

        # Volatility
        indicators['std_5'] = df[target].rolling(window=5, min_periods=5).std()
        indicators['std_10'] = df[target].rolling(window=10, min_periods=10).std()

        # RSI (classic 14-period)
        delta = df[target].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window=14, min_periods=14).mean()
        avg_loss = down.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        indicators['rsi_14'] = 100 - (100 / (1 + rs))

        return indicators

# %% [markdown]
