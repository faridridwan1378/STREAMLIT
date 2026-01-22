"""
IDX Stock Analyzer - V3 FIXED
Copyright © 2026 Farid Ridwan | farid.rdwan@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IDX Analyzer", page_icon="💰", layout="wide")

# Daftar Saham
IDX_STOCKS = {
    'BLUE_CHIP': ['BBCA', 'BBRI', 'BMRI', 'BBNI', 'TLKM', 'ASII', 'UNVR', 'ICBP', 'INDF', 'GGRM', 'KLBF', 'AMRT'],
    'LQ45': ['ADRO', 'ANTM', 'BRPT', 'CPIN', 'EXCL', 'GOTO', 'INKP', 'ITMG', 'JSMR', 'MDKA', 'MEDC', 'PGAS', 'PTBA', 'SMGR', 'TPIA', 'UNTR'],
    'BANKING': ['BBCA', 'BBRI', 'BMRI', 'BBNI', 'BRIS', 'BDMN', 'PNBN', 'MEGA'],
    'MINING': ['ADRO', 'ANTM', 'INCO', 'ITMG', 'PTBA', 'MEDC', 'HRUM', 'AKRA'],
    'TECH': ['TLKM', 'ISAT', 'EXCL', 'GOTO', 'BUKA', 'EMTK'],
    'CONSUMER': ['UNVR', 'ICBP', 'INDF', 'MYOR', 'CPIN', 'JPFA', 'SIDO'],
    'PROPERTY': ['BSDE', 'CTRA', 'SMRA', 'PWON', 'JSMR', 'PTPP', 'ADHI']
}

@st.cache_data(ttl=300)
def get_stock_data(ticker, period="2y"):
    if not ticker.endswith('.JK'):
        ticker = f"{ticker}.JK"
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert(None)
        
        df = df.set_index('Date')
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        cols = [c for c in cols if c in df.columns]
        return df[cols]
        
    except Exception as e:
        return pd.DataFrame()

def add_indicators(df):
    if df.empty or len(df) < 20:
        return df
    
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Direction'] = (df['Returns'] > 0).astype(int)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(min(50, len(df))).mean()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 0.0001)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA'] + 1)
    
    if isinstance(df.index, pd.DatetimeIndex):
        df['DayOfWeek'] = df.index.dayofweek
    
    return df.ffill().bfill()

def analyze_stats(df):
    results = {}
    if 'Returns' not in df.columns:
        return results
    
    returns = df['Returns'].dropna()
    total = len(returns)
    if total == 0:
        return results
    
    up = (returns > 0).sum()
    down = (returns < 0).sum()
    
    results['basic'] = {
        'total_days': total, 'up_days': int(up), 'down_days': int(down),
        'prob_up': up / total, 'prob_down': down / total,
        'avg_return': returns.mean(), 'std_return': returns.std()
    }
    
    if 'Direction' in df.columns:
        df_t = df.copy()
        df_t['Tomorrow'] = df_t['Direction'].shift(-1)
        df_t = df_t.dropna()
        up_t = df_t[df_t['Direction'] == 1]
        down_t = df_t[df_t['Direction'] == 0]
        results['conditional'] = {
            'prob_up_after_up': up_t['Tomorrow'].mean() if len(up_t) > 0 else 0,
            'prob_up_after_down': down_t['Tomorrow'].mean() if len(down_t) > 0 else 0
        }
    
    if 'DayOfWeek' in df.columns:
        days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat']
        day_s = df.groupby('DayOfWeek')['Direction'].agg(['mean', 'count'])
        day_s.columns = ['Prob_Naik', 'Jumlah']
        day_s.index = [days[i] if i < len(days) else f'Day{i}' for i in day_s.index]
        results['day_of_week'] = day_s
    
    return results

def get_recommendation(df, stats):
    if len(df) < 2:
        return "DATA KURANG", "❓", "#888", ["Butuh lebih banyak data"], 0
    
    last, prev = df.iloc[-1], df.iloc[-2]
    score = 0
    signals = []

    # RSI
    rsi = last.get('RSI', 50)
    if pd.notna(rsi):
        if rsi < 30:
            score += 3
            signals.append("🔥 RSI < 30: OVERSOLD")
        elif rsi < 40:
            score += 2
            signals.append("🟢 RSI Rendah: Akumulasi")
        elif rsi > 70:
            score -= 3
            signals.append("🛑 RSI > 70: OVERBOUGHT")
        elif rsi > 60:
            score -= 1
            signals.append("🟡 RSI Tinggi")
        else:
            signals.append(f"⚪ RSI: {rsi:.1f}")

    # MACD
    macd, sig = last.get('MACD', 0), last.get('MACD_Signal', 0)
    p_macd, p_sig = prev.get('MACD', 0), prev.get('MACD_Signal', 0)
    if pd.notna(macd):
        if macd > sig and p_macd <= p_sig:
            score += 4
            signals.append("🚀 GOLDEN CROSS MACD!")
        elif macd < sig and p_macd >= p_sig:
            score -= 4
            signals.append("⚠️ DEATH CROSS MACD!")
        elif macd > sig:
            score += 1
            signals.append("🟢 MACD Bullish")
        else:
            score -= 1
            signals.append("🔴 MACD Bearish")

    # Volume
    vol_r = last.get('Vol_Ratio', 1)
    ret = last.get('Returns', 0)
    if pd.notna(vol_r) and vol_r > 1.5:
        if ret > 0:
            score += 2
            signals.append("💪 Volume Tinggi + Naik")
        else:
            score -= 2
            signals.append("📉 Volume Tinggi + Turun")

    # Probability
    prob = stats.get('basic', {}).get('prob_up', 0.5) * 100
    if prob >= 55:
        score += 2
        signals.append(f"📊 Historis: {prob:.1f}% Naik")
    elif prob <= 45:
        score -= 1
        signals.append(f"📉 Historis: {prob:.1f}% Naik")

    # Trend
    c = last.get('Close', 0)
    s20, s50 = last.get('SMA_20', c), last.get('SMA_50', c)
    if c > s20 and c > s50:
        score += 2
        signals.append("📈 UPTREND")
    elif c < s20 and c < s50:
        score -= 2
        signals.append("📉 DOWNTREND")

    if score >= 6:
        return "💎 STRONG BUY", "🚀", "#00C853", signals, score
    elif score >= 2:
        return "✅ BUY", "📈", "#4CAF50", signals, score
    elif score <= -6:
        return "🚨 STRONG SELL", "⛔", "#D50000", signals, score
    elif score <= -2:
        return "📉 SELL", "📉", "#FF5722", signals, score
    else:
        return "⏳ HOLD", "⚖️", "#607D8B", signals, score

def plot_chart(df, ticker):
    if df.empty:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'), row=1, col=1)
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA20', line=dict(color='orange')), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA50', line=dict(color='blue')), row=1, col=1)
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, opacity=0.7, name='Vol'), row=2, col=1)
    fig.update_layout(title=f'📈 {ticker}', height=550, template='plotly_dark', xaxis_rangeslider_visible=False)
    return fig

def plot_rsi_macd(df, ticker):
    if df.empty:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.05, subplot_titles=['RSI', 'MACD'])
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='cyan')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
        colors = ['green' if v >= 0 else 'red' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='Hist'), row=2, col=1)
    fig.update_layout(height=500, template='plotly_dark')
    return fig

def main():
    st.markdown("""<style>
    .rec-box {padding:25px; border-radius:15px; text-align:center; color:white; margin:20px 0;}
    .sig {background:rgba(255,255,255,0.15); padding:8px 15px; border-radius:8px; margin:5px; text-align:left;}
    </style>""", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;'>💰 IDX Stock Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Analisis & Rekomendasi Saham Indonesia</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://raw.githubusercontent.com/faridridwan1378/varsha-catalog/main/gambar%20buah/gemini-2.5-flash-image-preview%20(nano-banana)_a_Create_an_ultra-real%20(3).png", use_container_width=True)
        st.markdown("---")
        cat = st.selectbox("📁 Kategori", list(IDX_STOCKS.keys()))
        ticker = st.selectbox("📊 Saham", IDX_STOCKS[cat])
        period = st.selectbox("📅 Periode", ["6mo", "1y", "2y", "5y"], index=1)
        st.markdown("---")
        btn = st.button("🔍 ANALISIS", use_container_width=True, type="primary")
        st.markdown("---")
        st.caption("**IDX Analyzer v3.0**")
        st.caption("© 2026 Farid Ridwan")
        st.caption("📧 farid.rdwan@gmail.com")

    if btn:
        with st.spinner(f"Mengambil data {ticker}..."):
            df_raw = get_stock_data(ticker, period)
        
        if df_raw.empty:
            st.error("❌ Gagal mengambil data!")
            return
        
        st.success(f"✅ {len(df_raw)} data {ticker}")
        
        df = add_indicators(df_raw)
        stats = analyze_stats(df)
        
        if df.empty:
            st.error("❌ Data tidak cukup")
            return
        
        status, icon, color, signals, score = get_recommendation(df, stats)
        
        sig_html = "".join([f'<div class="sig">{s}</div>' for s in signals])
        st.markdown(f"""
        <div class="rec-box" style="background:linear-gradient(135deg,{color},{color}99);">
            <h1 style="margin:0;font-size:2.5rem;">{icon} {status}</h1>
            <p style="font-size:1.2rem;margin:10px 0;">Score: <b style="background:rgba(255,255,255,0.3);padding:5px 15px;border-radius:20px;">{score:+d}</b></p>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;margin-top:15px;">{sig_html}</div>
        </div>""", unsafe_allow_html=True)

        last = df.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💵 Harga", f"Rp {last['Close']:,.0f}")
        c2.metric("📈 Change", f"{last['Returns']*100:+.2f}%")
        c3.metric("📊 RSI", f"{last['RSI']:.1f}")
        c4.metric("📉 MACD", f"{last['MACD']:.4f}")
        c5.metric("🎯 Prob", f"{stats.get('basic',{}).get('prob_up',0)*100:.1f}%")

        t1, t2, t3, t4 = st.tabs(["📈 Chart", "🔬 RSI/MACD", "📊 Stats", "📋 Data"])
        with t1:
            st.plotly_chart(plot_chart(df.tail(100), ticker), use_container_width=True)
        with t2:
            st.plotly_chart(plot_rsi_macd(df.tail(100), ticker), use_container_width=True)
        with t3:
            b = stats.get('basic', {})
            co = stats.get('conditional', {})
            col1, col2 = st.columns(2)
            col1.metric("📈 Prob Naik", f"{b.get('prob_up',0)*100:.1f}%", f"{b.get('up_days',0)} hari")
            col2.metric("📉 Prob Turun", f"{b.get('prob_down',0)*100:.1f}%", f"{b.get('down_days',0)} hari")
            st.markdown("---")
            st.write(f"• Naik setelah Naik: **{co.get('prob_up_after_up',0)*100:.1f}%**")
            st.write(f"• Naik setelah Turun: **{co.get('prob_up_after_down',0)*100:.1f}%**")
            if 'day_of_week' in stats:
                st.markdown("---")
                st.dataframe(stats['day_of_week'], use_container_width=True)
        with t4:
            cols = [c for c in ['Open','High','Low','Close','Volume','Returns','RSI','MACD'] if c in df.columns]
            st.dataframe(df[cols].tail(50), use_container_width=True)
            st.download_button("📥 CSV", df.to_csv(), f"{ticker}.csv")
    else:
        st.info("👋 Pilih saham dan klik **ANALISIS** untuk melihat rekomendasi")

    st.markdown("<hr><p style='text-align:center;color:#888;'>⚠️ Untuk kalangan sendiri. © 2026 Farid Ridwan | farid.rdwan@gmail.com</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()