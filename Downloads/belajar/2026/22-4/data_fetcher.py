"""
Data Fetcher Module for IDX Stocks
Copyright © 2026 Farid Ridwan | farid.rdwan@gmail.com
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class IDXDataFetcher:
    """Kelas untuk mengambil data saham IDX"""
    
    IDX_STOCKS = {
        'BLUE_CHIP': ['BBCA', 'BBRI', 'BMRI', 'BBNI', 'TLKM', 'ASII', 'UNVR', 'ICBP', 'INDF', 'GGRM', 'KLBF', 'AMRT'],
        'LQ45': ['ADRO', 'ANTM', 'BRPT', 'CPIN', 'EXCL', 'GOTO', 'INKP', 'ITMG', 'JSMR', 'MDKA', 'MEDC', 'PGAS', 'PTBA', 'SMGR', 'TPIA', 'UNTR'],
        'BANKING': ['BBCA', 'BBRI', 'BMRI', 'BBNI', 'BRIS', 'BDMN', 'PNBN', 'MEGA'],
        'MINING': ['ADRO', 'ANTM', 'INCO', 'ITMG', 'PTBA', 'MEDC', 'HRUM', 'AKRA'],
        'TECH': ['TLKM', 'ISAT', 'EXCL', 'GOTO', 'BUKA', 'EMTK'],
        'CONSUMER': ['UNVR', 'ICBP', 'INDF', 'MYOR', 'CPIN', 'JPFA', 'SIDO'],
        'PROPERTY': ['BSDE', 'CTRA', 'SMRA', 'PWON', 'JSMR', 'PTPP', 'ADHI']
    }
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Mengambil data saham dari Yahoo Finance"""
        if not ticker.endswith('.JK'):
            ticker = f"{ticker}.JK"
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d", progress=False)
            
            if df.empty:
                return pd.DataFrame()
            
            # PENTING: Reset index dan handle timezone
            df = df.reset_index()
            
            # Rename kolom pertama jadi 'Date' (bisa 'Date' atau 'Datetime')
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'Date'})
            
            # Convert ke datetime dan hapus timezone
            df['Date'] = pd.to_datetime(df['Date'])
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_convert(None)
            
            # Set index
            df = df.set_index('Date')
            
            # Pastikan kolom standar ada
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    # Cari dengan case-insensitive
                    for c in df.columns:
                        if c.lower() == col.lower():
                            df = df.rename(columns={c: col})
                            break
            
            # Ambil hanya kolom yang dibutuhkan
            cols = [c for c in required_cols if c in df.columns]
            df = df[cols]
            
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_available_stocks() -> dict:
        return IDXDataFetcher.IDX_STOCKS

    def get_stock_info(self, ticker: str) -> dict:
        return {'name': ticker, 'sector': 'IDX Stock'}


class FeatureEngineer:
    """Kelas untuk membuat indikator teknikal"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 50:
            return df
            
        df = df.copy()
        
        # 1. Returns
        df['Returns'] = df['Close'].pct_change()
        df['Direction'] = (df['Returns'] > 0).astype(int)
        
        # 2. Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # 3. RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 0.0001)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 5. Volume Analysis
        df['Vol_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA'] + 1)  # Avoid division by zero
        
        # 6. Bollinger Bands
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)
        
        # 7. Day of Week & Month
        if isinstance(df.index, pd.DatetimeIndex):
            df['DayOfWeek'] = df.index.dayofweek
            df['Month'] = df.index.month
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df