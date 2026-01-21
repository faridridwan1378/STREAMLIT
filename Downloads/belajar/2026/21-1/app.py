import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# --- KONFIGURASI HALAMAN ---
st.set_page_config(layout="wide", page_title="Saham Analyzer Pro", page_icon="📈")

# --- SIDEBAR ---
st.sidebar.header('⚙️ Konfigurasi')
ticker_input = st.sidebar.text_input('Daftar Saham (pisahkan koma)', 'BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK')
# Membersihkan input list
tickers = [x.strip().upper() for x in ticker_input.split(',') if x.strip() != '']

start_date = st.sidebar.date_input('Mulai', datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input('Akhir', datetime.now())

if st.sidebar.button('Refresh Data'):
    st.cache_data.clear()

# --- FUNGSI UTAMA ---
def get_data(ticker_list, start, end):
    """
    Mengambil data dengan auto_adjust=True.
    Ini berarti 'Close' sudah memperhitungkan Dividen/Split.
    Tidak ada lagi kolom 'Adj Close'.
    """
    try:
        data = yf.download(ticker_list, start=start, end=end, auto_adjust=True, threads=True)
        return data
    except Exception as e:
        st.error(f"Gagal download data: {e}")
        return pd.DataFrame()

# --- LOAD DATA ---
st.title('📈 Dashboard Saham Interaktif')
data_state = st.empty()
data_state.text('Memuat data...')

try:
    # Download data sekaligus
    raw_data = get_data(tickers, start_date, end_date)
    
    # Cek apakah data kosong
    if raw_data.empty:
        st.error("Data kosong. Periksa koneksi internet atau kode saham.")
        st.stop()

    data_state.text('') # Hapus loading text

    # --- MEMPERSIAPKAN DATAFRAME KHUSUS CLOSE ---
    # Menangani kasus 1 saham vs banyak saham
    if len(tickers) == 1:
        # Jika cuma 1 saham, yfinance mengembalikan DataFrame flat. 
        # Kita ubah jadi DataFrame dengan nama kolom saham tersebut agar konsisten.
        df_close = raw_data[['Close']].copy()
        df_close.columns = tickers
    else:
        # Jika banyak saham, ambil level 'Close'
        # yfinance terbaru mungkin mengembalikan MultiIndex (Price, Ticker) atau (Ticker, Price)
        # Kita coba ambil 'Close' dengan aman
        try:
            df_close = raw_data['Close']
        except KeyError:
            # Fallback jika struktur berbeda
            df_close = raw_data.xs('Close', level=0, axis=1)

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs(["📊 Perbandingan", "🔍 Teknikal", "📋 Fundamental"])

    # --- TAB 1: PERBANDINGAN ---
    with tab1:
        st.subheader("Perbandingan Kinerja (%)")
        
        # Hapus baris yang semuanya NaN (hari libur bursa)
        df_close_clean = df_close.dropna(how='all')
        
        if not df_close_clean.empty:
            # Normalisasi (Return sejak hari pertama dipilih)
            normalized = ((df_close_clean / df_close_clean.iloc[0]) - 1) * 100
            
            fig_comp = px.line(normalized, x=normalized.index, y=normalized.columns,
                               labels={'value': 'Keuntungan (%)', 'variable': 'Kode Saham', 'Date': 'Tanggal'})
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Tabel Performa
            last_val = normalized.iloc[-1].sort_values(ascending=False)
            st.write("Top Gainers (Periode Ini):")
            st.dataframe(last_val.to_frame("Return %").style.format("{:.2f}%"))
        else:
            st.warning("Data harga tidak tersedia untuk periode ini.")

    # --- TAB 2: TEKNIKAL ---
    with tab2:
        col_t1, col_t2 = st.columns([1, 3])
        
        with col_t1:
            selected_ticker = st.selectbox("Pilih Saham", tickers)
            st.caption("Pilih Indikator:")
            check_sma = st.checkbox("SMA (50 & 200)", value=True)
            check_bb = st.checkbox("Bollinger Bands")
            check_rsi = st.checkbox("RSI")
            check_macd = st.checkbox("MACD")

        with col_t2:
            # Ambil data spesifik saham dari raw_data
            if len(tickers) == 1:
                df_single = raw_data.copy()
            else:
                # Mengambil data untuk ticker spesifik dari MultiIndex
                # Cara paling aman di pandas terbaru untuk swaplevel
                try:
                    df_single = raw_data.xs(selected_ticker, axis=1, level=1)
                except:
                    # Coba cara lain jika level tertukar
                    df_single = raw_data.xs(selected_ticker, axis=1, level=0)

            # Pastikan drop NA agar indikator akurat
            df_single = df_single.dropna()

            if not df_single.empty:
                # --- CHART UTAMA ---
                fig = go.Figure()
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df_single.index,
                    open=df_single['Open'], high=df_single['High'],
                    low=df_single['Low'], close=df_single['Close'],
                    name='Price'
                ))

                # Indikator SMA
                if check_sma:
                    sma50 = SMAIndicator(close=df_single['Close'], window=50).sma_indicator()
                    sma200 = SMAIndicator(close=df_single['Close'], window=200).sma_indicator()
                    fig.add_trace(go.Scatter(x=df_single.index, y=sma50, line=dict(color='orange', width=1), name='SMA 50'))
                    fig.add_trace(go.Scatter(x=df_single.index, y=sma200, line=dict(color='blue', width=1), name='SMA 200'))

                # Indikator BB
                if check_bb:
                    bb = BollingerBands(close=df_single['Close'], window=20, window_dev=2)
                    fig.add_trace(go.Scatter(x=df_single.index, y=bb.bollinger_hband(), line=dict(color='gray', width=1, dash='dot'), name='BB Up'))
                    fig.add_trace(go.Scatter(x=df_single.index, y=bb.bollinger_lband(), line=dict(color='gray', width=1, dash='dot'), name='BB Low'))

                fig.update_layout(title=f"Analisa Teknikal {selected_ticker}", height=500, yaxis_title="Harga", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # --- INDIKATOR BAWAH ---
                if check_rsi:
                    rsi = RSIIndicator(close=df_single['Close']).rsi()
                    fig_rsi = px.line(x=df_single.index, y=rsi, title="RSI Momentum")
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(height=250)
                    st.plotly_chart(fig_rsi, use_container_width=True)

                if check_macd:
                    macd = MACD(close=df_single['Close'])
                    df_macd = pd.DataFrame({'MACD': macd.macd(), 'Signal': macd.macd_signal()}, index=df_single.index)
                    fig_macd = px.line(df_macd, title="MACD Trend")
                    fig_macd.update_layout(height=250)
                    st.plotly_chart(fig_macd, use_container_width=True)
            else:
                st.warning("Data tidak cukup untuk menampilkan grafik.")

    # --- TAB 3: FUNDAMENTAL ---
    with tab3:
        st.subheader(f"Info Perusahaan: {selected_ticker}")
        try:
            info = yf.Ticker(selected_ticker).info
            # Filter info penting saja
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                st.metric("Harga Terakhir", info.get('currentPrice', info.get('regularMarketPreviousClose', '-')))
                st.metric("Sektor", info.get('sector', '-'))
            
            with col_f2:
                mk_cap = info.get('marketCap', 0)
                st.metric("Market Cap", f"{mk_cap:,.0f}" if mk_cap else "-")
                st.metric("PE Ratio", info.get('trailingPE', '-'))

            with col_f3:
                div = info.get('dividendYield', 0)
                st.metric("Dividend Yield", f"{div*100:.2f}%" if div else "-")
                st.metric("Rekomendasi", info.get('recommendationKey', '-').upper())
            
            with st.expander("Baca Deskripsi Bisnis"):
                st.write(info.get('longBusinessSummary', 'Tidak ada deskripsi.'))
                
        except Exception as e:
            st.warning(f"Gagal mengambil data fundamental: {e}")

except Exception as e:
    st.error(f"Terjadi kesalahan fatal: {e}")
    st.write("Saran: Coba refresh halaman atau ganti kode saham.")