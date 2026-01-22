"""
Visualization Module
Copyright © 2026 Farid Ridwan | farid.rdwan@gmail.com
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class StockVisualizer:
    """Kelas untuk visualisasi data saham"""
    
    @staticmethod
    def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
        """Membuat candlestick chart dengan volume"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No Data Available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # SMA
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], 
                          name='SMA 20', line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], 
                          name='SMA 50', line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] 
                 else 'green' for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                  marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'📈 {ticker} - Price Chart',
            xaxis_rangeslider_visible=False,
            height=600,
            template='plotly_dark',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_indicators(df: pd.DataFrame, ticker: str) -> go.Figure:
        """Chart dengan RSI dan MACD"""
        if df.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price', 'RSI', 'MACD')
        )
        
        # Price
        fig.add_trace(
            go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                          low=df['Low'], close=df['Close'], name='Price'),
            row=1, col=1
        )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='cyan')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')),
                row=3, col=1
            )
            
            colors = ['green' if v >= 0 else 'red' for v in df['MACD_Hist']]
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=colors),
                row=3, col=1
            )
        
        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig