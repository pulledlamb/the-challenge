# %%
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import datetime as dt
from urllib.parse import urlparse
import os


class DataProcessor:
    def __init__(self, ticker_dict, fred_api_key, start_date="2011-01-04"):
        self.ticker_dict = ticker_dict
        self.start_date = start_date
        self.fred = Fred(api_key=fred_api_key)
        self.data = None

    # ---------- Yahoo Finance ----------
    def download_yf_data(self):
        data = {}
        for ticker in self.ticker_dict:
            try:
                df = yf.download(ticker, start=self.start_date)
                if not df.empty:
                    data[ticker] = df['Close']
                    # if containing na values, fill them 
                    print(f"✓ {ticker}: {len(df)} records")
                else:
                    print(f"✗ {ticker}: No data")
            except Exception as e:
                print(f"✗ {ticker}: Error - {e}")

        combined = pd.concat(data, axis=1)
        combined['date'] = combined.index
        cols = [self.ticker_dict[t] for t in self.ticker_dict]
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
    def load_rba_rate(self, url):
        ir_au = self.load_data_from_url(url)
        ir_au = ir_au.rename(columns={
            "F1 INTEREST RATES AND YIELDS – MONEY MARKET": "date",
            "Unnamed: 1": "rates"
        }).iloc[10:][['date', 'rates']]
        ir_au['date'] = pd.to_datetime(ir_au['date'])
        ir_au['rates'] = pd.to_numeric(ir_au['rates'])
        ir_au = ir_au[ir_au['date'] >= pd.to_datetime(self.start_date)]
        return ir_au.reset_index(drop=True)

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
        return df_fed

    # ---------- CPI ----------
    def load_cpi(self, series_id, col_name):
        cpi = self.fred.get_series(series_id, self.start_date, dt.datetime.now())
        df_cpi = pd.DataFrame(cpi).reset_index()
        df_cpi.columns = ['date', col_name]
        df_cpi['date'] = pd.to_datetime(df_cpi['date'])
        df_cpi[col_name] = pd.to_numeric(df_cpi[col_name])
        return df_cpi

    # ---------- Merge all ----------
    def merge_all(self, rba_url):
        rba = self.load_rba_rate(rba_url)
        fed = self.load_fed_rate()
        au_cpi = self.load_cpi('FPCPITOTLZGAUS', 'cpi')
        us_cpi = self.load_cpi('FPCPITOTLZGUSA', 'us_cpi')

        df = self.data.merge(rba, on='date', how='left')
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


# %% [markdown]
# Example usage
# %%
ticker_dict = {
    'AUDUSD=X': 'audusd',
    'TIO=F': 'iron_ore',
    'GC=F': 'gold',
    '^VIX': 'vix',
    'USDJPY=X': 'usdjpy',
    'USDCNY=X': 'usdcny',
    'BTC-USD': 'btc',
}

end = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
rba_url = f"https://www.rba.gov.au/statistics/tables/xls/f01d.xlsx?v={end}"
dp = DataProcessor(ticker_dict, fred_api_key='28285699c4f4b42ddfeeb266781312d1')

dp.download_yf_data()
dp.merge_all(rba_url)
dp.export()

df = pd.read_parquet('data.parquet')
print(df.head())
df = df.ffill().bfill()
print(df.isna().sum())  # Check for missing values
