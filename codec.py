import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Analizador de Trades OKX")
BASE_DIR = "datos_historicos"

# Inicializar Session State para controlar la carga de datos
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.moneda = None
    st.session_state.trade_id = None
    st.session_state.global_start = None
    st.session_state.global_end = None


# ==========================================
# 0. GESTOR DE ARCHIVOS (Sin cambios)
# ==========================================
def gestionar_directorio(symbol):
    safe_symbol = symbol.replace(":", "-").replace("/", "-")
    ruta_dir = os.path.join(BASE_DIR, safe_symbol)
    if not os.path.exists(ruta_dir):
        os.makedirs(ruta_dir)
    return ruta_dir

def guardar_datos_csv(df, symbol, tipo_dato):
    ruta_dir = gestionar_directorio(symbol)
    ruta_archivo = os.path.join(ruta_dir, f"{tipo_dato}.csv")
    df.to_csv(ruta_archivo, index=False)
    return len(df)

def cargar_datos_csv(symbol, tipo_dato):
    ruta_dir = gestionar_directorio(symbol)
    ruta_archivo = os.path.join(ruta_dir, f"{tipo_dato}.csv")
    if os.path.exists(ruta_archivo):
        try:
            df = pd.read_csv(ruta_archivo)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def get_current_price(symbol):
    try:
        exchange = ccxt.okx()
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception:
        return None

# ==========================================
# 1. FUNCIONES MATEMÁTICAS (Sin cambios)
# ==========================================
def calculate_rsi_series(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_supertrend(df, period=10, multiplier=3):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    final_upper = pd.Series(np.nan, index=df.index)
    final_lower = pd.Series(np.nan, index=df.index)
    trend = pd.Series(1, index=df.index)
    
    if len(df) > 0:
        final_upper.iloc[0] = basic_upper.iloc[0]
        final_lower.iloc[0] = basic_lower.iloc[0]

    for i in range(1, len(df)):
        if np.isnan(basic_upper[i]): final_upper[i] = final_upper[i-1]
        elif basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]: final_upper[i] = basic_upper[i]
        else: final_upper[i] = final_upper[i-1]
        
        if np.isnan(basic_lower[i]): final_lower[i] = final_lower[i-1]
        elif basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]: final_lower[i] = basic_lower[i]
        else: final_lower[i] = final_lower[i-1]
        
        if trend[i-1] == 1:
            if close[i] < final_lower[i]: trend[i] = -1
            else: trend[i] = 1
        else:
            if close[i] > final_upper[i]: trend[i] = 1
            else: trend[i] = -1
    return trend, final_upper, final_lower

def calculate_stochrsi(df, period=14, smoothK=3, smoothD=3):
    rsi = calculate_rsi_series(df['close'], period)
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    stoch = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    k = stoch.rolling(window=smoothK).mean()
    d = k.rolling(window=smoothD).mean()
    return k, d

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger(df, period=20, std_dev=2):
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower, sma

# ==========================================
# 2. MOTOR VELAS (Sin cambios)
# ==========================================
@st.cache_data(ttl=300) 
def descargar_velas_recientes_api(symbol, timeframe):
    try:
        headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar={timeframe}&limit=100"
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if data.get('code') == '0' and data.get('data'):
            return data['data']
        return []
    except Exception:
        return []

def descargar_tramo_api(symbol, timeframe, start_ts_ms, end_cursor_ms):
    all_data = []
    after_cursor = f"&after={end_cursor_ms}"
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    prog_text = st.sidebar.empty()
    try:
        for i in range(300):
            url = f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar={timeframe}&limit=100{after_cursor}"
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            if data['code'] == '0' and data['data']:
                batch = data['data']
                all_data.extend(batch)
                last_ts = int(batch[-1][0])
                f_str = datetime.fromtimestamp(last_ts/1000).strftime('%d-%b %H:%M')
                prog_text.text(f"📉 Velas: {f_str}")
                if last_ts < start_ts_ms: break
                after_cursor = f"&after={last_ts}"
                time.sleep(0.1)
            else: break
        prog_text.empty()
        return all_data
    except: return []

def procesar_lista_velas(lista_raw):
    if not lista_raw: return pd.DataFrame()
    df = pd.DataFrame(lista_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def obtener_velas_smart(symbol, start_date, end_date, timeframe='15m'):
    tf_value = int(timeframe.replace('m', ''))
    tf_as_delta = timedelta(minutes=tf_value)

    df_local = cargar_datos_csv(symbol, 'velas')
    req_start = pd.to_datetime(start_date)
    req_end = pd.to_datetime(end_date) + timedelta(days=1)
    req_start_ms = int(req_start.timestamp() * 1000)
    req_end_ms = int(datetime.utcnow().timestamp() * 1000)
    cambios = False

    if df_local.empty:
        st.toast(f"📥 Descargando historial {symbol}...")
        raw = descargar_tramo_api(symbol, timeframe, req_start_ms, req_end_ms)
        df_local = procesar_lista_velas(raw)
        cambios = True
    else:
        l_min = df_local['timestamp'].min()
        l_max = df_local['timestamp'].max()
        l_min_ms = int(l_min.timestamp() * 1000)
        
        if req_start < l_min:
            st.toast(f"⏪ Cargando pasado ({start_date})...")
            raw_past = descargar_tramo_api(symbol, timeframe, req_start_ms, l_min_ms)
            if raw_past:
                df_past = procesar_lista_velas(raw_past)
                df_local = pd.concat([df_past, df_local])
                cambios = True

        if datetime.utcnow() > l_max + tf_as_delta:
            st.toast("⏩ Actualizando reciente...")
            raw_future = descargar_velas_recientes_api(symbol, timeframe)
            if raw_future:
                df_fut = procesar_lista_velas(raw_future)
                df_local = pd.concat([df_local, df_fut])
                cambios = True

    if cambios and not df_local.empty:
        df_local = df_local.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        guardar_datos_csv(df_local, symbol, 'velas')

    if not df_local.empty:
        mask = (df_local['timestamp'] >= req_start) & (df_local['timestamp'] <= req_end)
        return df_local.loc[mask].sort_values('timestamp').reset_index(drop=True)
    return pd.DataFrame()

# ==========================================
# 3. MOTOR OI (Sin cambios)
# ==========================================
def descargar_oi_rango(symbol, start_date, end_date):
    ts_start_meta = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1000)
    ts_end_limit = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1000)
    
    all_oi_data = []
    current_end_cursor = f"&end={ts_end_limit}" 
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    
    status_box = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    log_box = st.sidebar.expander("📜 Log Técnico OI", expanded=True)
    
    max_loops = 300 
    last_ts_seen = float('inf')
    
    try:
        for i in range(max_loops):
            url = f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history?instId={symbol}&period=5m&limit=100{current_end_cursor}"
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            
            if data['code'] == '0' and data['data']:
                batch = data['data']
                oldest_ts = int(batch[-1][0])
                
                if oldest_ts >= last_ts_seen:
                    log_box.warning(f"⚠️ Bucle en {oldest_ts}. Parando.")
                    break
                last_ts_seen = oldest_ts
                
                all_oi_data.extend(batch)
                
                f_lote = datetime.fromtimestamp(oldest_ts/1000).strftime('%d-%b %H:%M')
                log_box.write(f"Lote {i+1}: {len(batch)} datos. Hasta: {f_lote}")
                
                total = ts_end_limit - ts_start_meta
                curr = ts_end_limit - oldest_ts
                pct = min(max(curr/total, 0.0), 1.0) if total > 0 else 0
                progress_bar.progress(pct)

                if oldest_ts < ts_start_meta:
                    progress_bar.progress(1.0)
                    status_box.success(f"✅ Descargados {len(all_oi_data)} registros.")
                    break
                
                current_end_cursor = f"&end={oldest_ts}"
                time.sleep(0.15) 
            else:
                log_box.error(f"API Stop: {data}")
                break
        
        status_box.empty()
        progress_bar.empty()
        
        if not all_oi_data: return pd.DataFrame()
        
        df_oi = pd.DataFrame(all_oi_data, columns=['timestamp', 'oi_contracts', 'openInterest', 'oi_usd'])
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
        df_oi['openInterest'] = pd.to_numeric(df_oi['openInterest'], errors='coerce')
        
        mask = (df_oi['timestamp'] >= pd.to_datetime(ts_start_meta, unit='ms')) & \
               (df_oi['timestamp'] <= pd.to_datetime(ts_end_limit, unit='ms'))
        
        return df_oi.loc[mask].sort_values('timestamp')

    except Exception as e:
        st.sidebar.error(f"Error crítico OI: {e}")
        return pd.DataFrame()

def obtener_oi_smart(symbol, start_date, end_date):
    symbol_oi = symbol if "-SWAP" in symbol else f"{symbol}-SWAP"
    df_local = cargar_datos_csv(symbol_oi, 'open_interest')
    req_start = pd.to_datetime(start_date)
    req_end = pd.to_datetime(end_date) + timedelta(days=1)
    if not df_local.empty:
        mask = (df_local['timestamp'] >= req_start) & (df_local['timestamp'] <= req_end)
        return df_local.loc[mask].sort_values('timestamp')
    return pd.DataFrame()

@st.cache_data
def cargar_datos_usuario(archivo):
    try:
        if archivo.name.endswith('.csv'): df = pd.read_csv(archivo)
        else: df = pd.read_excel(archivo)
        cols = ['openAvgPx', 'closeAvgPx', 'pnl', 'closePnl']
        for col in cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except: return pd.DataFrame()

# ==========================================
# 4. GRAFICADO (FUNCIÓN CACHEADA - Sin cambios)
# ==========================================
@st.cache_data
def generate_chart(df_velas, trade, moneda_selec, offset_horas, show_ema_9, show_ema_21, show_ema_20, show_ema_50, show_ema_200, show_macd, show_rsi, show_stochrsi, show_bb, show_supertrend, show_volume, show_oi):
    df_velas['timestamp'] = df_velas['timestamp'] + pd.Timedelta(hours=offset_horas)
    
    df_velas['EMA_9'] = df_velas['close'].ewm(span=9).mean()
    df_velas['EMA_21'] = df_velas['close'].ewm(span=21).mean()
    df_velas['EMA_20'] = df_velas['close'].ewm(span=20).mean()
    df_velas['EMA_50'] = df_velas['close'].ewm(span=50).mean()
    df_velas['EMA_200'] = df_velas['close'].ewm(span=200).mean()
    
    st_trend, st_upper, st_lower = calculate_supertrend(df_velas)
    df_velas['Supertrend'] = np.where(st_trend==1, st_lower, st_upper)
    df_velas['ST_Trend'] = st_trend
    
    k, d = calculate_stochrsi(df_velas)
    df_velas['K'], df_velas['D'] = k, d
    
    df_velas['RSI'] = calculate_rsi_series(df_velas['close'])
    m, s, h = calculate_macd(df_velas)
    df_velas['MACD'], df_velas['Signal'], df_velas['Hist'] = m, s, h
    bbu, bbl, _ = calculate_bollinger(df_velas)
    df_velas['BBU'], df_velas['BBL'] = bbu, bbl

    paneles = []
    if show_volume: paneles.append({'title': 'Volumen', 'type': 'vol'})
    if show_oi and 'openInterest' in df_velas.columns: 
        paneles.append({'title': 'Open Interest', 'type': 'oi'})
    if show_stochrsi: paneles.append({'title': 'Stoch RSI', 'type': 'stoch'})
    if show_rsi: paneles.append({'title': 'RSI', 'type': 'rsi'})
    if show_macd: paneles.append({'title': 'MACD', 'type': 'macd'})
    
    rows = 1 + len(paneles)
    row_h = [0.5] + [0.5/len(paneles)]*len(paneles) if paneles else [1.0]
    titles = [f"{moneda_selec}"] + [p['title'] for p in paneles]
    
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, 
        vertical_spacing=0.02, row_heights=row_h, subplot_titles=titles
    )

    fig.add_trace(go.Candlestick(x=df_velas['timestamp'], open=df_velas['open'], high=df_velas['high'], low=df_velas['low'], close=df_velas['close'], name='Precio'), row=1, col=1)
    
    if show_ema_9: fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['EMA_9'], line=dict(color='yellow', width=1), name='EMA 9'), row=1, col=1)
    if show_ema_21: fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['EMA_21'], line=dict(color='cyan', width=1), name='EMA 21'), row=1, col=1)
    if show_ema_20: fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
    if show_ema_50: fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['EMA_50'], line=dict(color='white', width=1), name='EMA 50'), row=1, col=1)
    if show_ema_200: fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['EMA_200'], line=dict(color='#9c27b0', width=2), name='EMA 200'), row=1, col=1)
    
    if show_bb:
        fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['BBU'], line=dict(color='gray', width=1), name='BBU'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['BBL'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='BBL'), row=1, col=1)

    if show_supertrend:
        st_green = df_velas['Supertrend'].copy()
        st_red = df_velas['Supertrend'].copy()
        st_green.loc[df_velas['ST_Trend'] != 1] = np.nan
        st_red.loc[df_velas['ST_Trend'] != -1] = np.nan
        for i in range(1, len(df_velas)):
            if df_velas['ST_Trend'].iloc[i-1] == 1 and df_velas['ST_Trend'].iloc[i] == -1: st_red.iloc[i-1] = df_velas['Supertrend'].iloc[i-1]
            elif df_velas['ST_Trend'].iloc[i-1] == -1 and df_velas['ST_Trend'].iloc[i] == 1: st_green.iloc[i-1] = df_velas['Supertrend'].iloc[i-1]
        fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=st_green, mode='lines', line=dict(color='#00c853', width=2), name='ST Buy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=st_red, mode='lines', line=dict(color='#ff5252', width=2), name='ST Sell'), row=1, col=1)

    curr_row = 2
    for p in paneles:
        if p['type'] == 'vol':
            c = ['#ef5350' if cl < op else '#26a69a' for cl, op in zip(df_velas['close'], df_velas['open'])]
            fig.add_trace(go.Bar(x=df_velas['timestamp'], y=df_velas['volume'], marker_color=c, name='Vol'), row=curr_row, col=1)
        elif p['type'] == 'oi':
            fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['openInterest'], mode='lines', line=dict(color='#2962ff', width=2), name='OI'), row=curr_row, col=1)
        elif p['type'] == 'stoch':
            fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['K'], line=dict(color='#2962ff', width=1), name='K'), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['D'], line=dict(color='#ff6d00', width=1), name='D'), row=curr_row, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="gray", row=curr_row, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="gray", row=curr_row, col=1)
        elif p['type'] == 'rsi':
            fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['RSI'], line=dict(color='#ba68c8', width=1.5), name='RSI'), row=curr_row, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=curr_row, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=curr_row, col=1)
        elif p['type'] == 'macd':
            c = ['#26a69a' if h >= 0 else '#ef5350' for h in df_velas['Hist']]
            fig.add_trace(go.Bar(x=df_velas['timestamp'], y=df_velas['Hist'], marker_color=c, name='Hist'), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['MACD'], line=dict(color='#2962ff', width=1), name='MACD'), row=curr_row, col=1)
            fig.add_trace(go.Scatter(x=df_velas['timestamp'], y=df_velas['Signal'], line=dict(color='#ff6d00', width=1), name='Sig'), row=curr_row, col=1)
        curr_row += 1

    # Trade
    px_in, px_out = float(trade['openAvgPx']), float(trade['closeAvgPx'])
    ts_in = pd.to_datetime(trade['openTime'])
    
    if px_out > 0:
        try: ts_out = pd.to_datetime(trade['uTime'])
        except: ts_out = df_velas['timestamp'].iloc[-1]
        c_trade = '#26a69a' if float(trade['pnl']) >= 0 else '#ef5350'
        is_open = False
    else:
        ts_out = df_velas['timestamp'].iloc[-1]
        px_out = df_velas['close'].iloc[-1]
        c_trade = '#2962ff'
        is_open = True

    fig.add_trace(go.Scatter(x=[ts_in, ts_out], y=[px_in, px_out], mode="lines+markers", line=dict(color=c_trade, width=2), showlegend=False, hoverinfo='skip'), row=1, col=1)
    
    fig.add_hline(y=px_in, line_dash="dot", line_color='#9c27b0', line_width=1, row=1, col=1)
    fig.add_vline(x=ts_in, line_dash="dot", line_color='#9c27b0', line_width=1)
    fig.add_annotation(x=1, y=px_in, xref="paper", yref="y", text=f"{px_in:.4f}", showarrow=False, bgcolor='#9c27b0', font=dict(color="white", size=11), xanchor="left", xshift=2)
    fig.add_annotation(x=ts_in, y=0, xref="x", yref="y domain", text=ts_in.strftime('%d %b %H:%M'), showarrow=False, bgcolor='#9c27b0', font=dict(color="white", size=10), yanchor="top", yshift=-5)

    if not is_open:
        fig.add_hline(y=px_out, line_dash="dot", line_color=c_trade, line_width=1, row=1, col=1)
        fig.add_vline(x=ts_out, line_dash="dot", line_color=c_trade, line_width=1)
        fig.add_annotation(x=1, y=px_out, xref="paper", yref="y", text=f"{px_out:.4f}", showarrow=False, bgcolor=c_trade, font=dict(color="white", size=11), xanchor="left", xshift=2)
        fig.add_annotation(x=ts_out, y=0, xref="x", yref="y domain", text=ts_out.strftime('%d %b %H:%M'), showarrow=False, bgcolor=c_trade, font=dict(color="white", size=10), yanchor="top", yshift=-25)

    total_height = 600 + (len(paneles) * 150)
    fig.update_layout(
        title=f"{moneda_selec}", template="plotly_dark", height=total_height,
        xaxis_rangeslider_visible=False, margin=dict(r=100, b=40, t=50, l=20),
        hovermode='x unified'
    )
    fig.update_yaxes(side="right", showgrid=False, gridcolor="#444", zeroline=False, fixedrange=False)
    fig.update_xaxes(showgrid=False, gridcolor="#444", fixedrange=False)
    
    return fig

# ==========================================
# INTERFAZ DE USUARIO (SIDEBAR)
# ==========================================
st.title("📊 Visualizador Pro (Sistema de Archivos)")

# --- CONFIGURACIÓN SIDEBAR ---
st.sidebar.header("1. Datos")
archivo_subido = st.sidebar.file_uploader("Sube tu Excel/CSV", type=['xlsx', 'csv'])

st.sidebar.markdown("---")
st.sidebar.header("2. Sincronización")
offset_horas = st.sidebar.number_input("Ajuste Horario", min_value=-12, max_value=12, value=-5, step=1)

st.sidebar.markdown("---")
with st.sidebar.expander("📈 3. Indicadores", expanded=True):
    c1, c2 = st.columns(2)
    show_ema_9 = c1.checkbox("EMA 9", True)
    show_ema_21 = c2.checkbox("EMA 21", True)
    show_ema_20 = c1.checkbox("EMA 20", False)
    show_ema_50 = c2.checkbox("EMA 50", False)
    show_ema_200 = c1.checkbox("EMA 200", False)
    c3, c4 = st.columns(2)
    show_macd = c3.checkbox("MACD", False)
    show_rsi = c4.checkbox("RSI", False)
    show_stochrsi = c3.checkbox("Stoch RSI", True)
    c5, c6 = st.columns(2)
    show_bb = c5.checkbox("Bollinger", False)
    show_supertrend = c6.checkbox("Supertrend", True)
    show_volume = c5.checkbox("Volumen", True)
    show_oi = c6.checkbox("Open Interest", True)

# ==========================================
# LÓGICA PRINCIPAL
# ==========================================

# Cargar datos de trades primero
if archivo_subido is not None:
    df_trades = cargar_datos_usuario(archivo_subido)
    if df_trades.empty or 'instId' not in df_trades.columns:
        st.error("Archivo no válido o no contiene la columna 'instId'.")
        st.stop()
    
    # --- ¡NUEVO! (Petición 1): Convertir ID a string para text_input ---
    df_trades['id'] = df_trades['id'].astype(str)

else:
    st.info("Por favor, sube tu archivo de trades para comenzar.")
    st.stop()

# --- 1. DECLARAR FILTROS (Fijos en la parte superior) ---
monedas = df_trades['instId'].astype(str).unique()

col_m1, col_m2 = st.columns(2)
with col_m1: 
    default_moneda_index = 0
    if st.session_state.moneda:
        try:
            default_moneda_index = int(np.where(monedas == st.session_state.moneda)[0][0])
        except: pass
    moneda_selec = st.selectbox("Moneda", monedas, index=default_moneda_index)

# (Petición 2: Filtro dependiente) - Esto ya estaba implementado
trades_moneda = df_trades[df_trades['instId'] == moneda_selec].copy()

with col_m2:
    # --- ¡MODIFICADO! (Petición 1): Cambiado a text_input ---
    trade_ids_list = trades_moneda['id'].tolist()
    default_trade_id_str = "" # Default vacío
    
    if st.session_state.trade_id and st.session_state.trade_id in trade_ids_list:
        # Si el ID guardado en sesión es válido para esta moneda, usarlo
        default_trade_id_str = st.session_state.trade_id
    elif trade_ids_list:
        # Si no, y la lista no está vacía, usar el primer ID de la lista
        default_trade_id_str = trade_ids_list[0]
    
    trade_id_input = st.text_input("Trade ID (copiar/pegar)", value=default_trade_id_str)
    # --- Fin de la modificación ---

# Pre-rellenar fechas basado en el trade
try:
    # Usamos el 'trade_id_input' para encontrar la fecha
    trade_obj = trades_moneda[trades_moneda['id'] == trade_id_input].iloc[0]
    default_trade_date = pd.to_datetime(trade_obj['openTime'])
    if pd.isna(default_trade_date): default_trade_date = datetime.now()
except Exception:
    default_trade_date = datetime.now()

default_start = st.session_state.global_start or (default_trade_date - timedelta(days=3)).date()
default_end = st.session_state.global_end or (default_trade_date + timedelta(days=1)).date()

col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
with col_f1: 
    global_start = st.date_input("Desde", default_start)
with col_f2: 
    global_end = st.date_input("Hasta", default_end)
with col_f3:
    st.markdown("<br>", unsafe_allow_html=True) # Espaciador
    apply_button = st.button("Aplicar Filtros", use_container_width=True)

# --- 2. DECLARAR PLACEHOLDERS (Debajo de los filtros) ---
header_placeholder = st.empty()
chart_placeholder = st.empty()

# --- LÓGICA DE CARGA (Tras presionar el botón) ---
if apply_button:
    st.session_state.moneda = moneda_selec
    st.session_state.trade_id = trade_id_input # Guardamos el valor del text_input
    st.session_state.global_start = global_start
    st.session_state.global_end = global_end
    st.session_state.data_loaded = True

if st.session_state.data_loaded:
    # Recuperar valores del state para consistencia
    moneda = st.session_state.moneda
    trade_id_state = st.session_state.trade_id # Este es un string
    start_date = st.session_state.global_start
    end_date = st.session_state.global_end
    
    try:
        trade = df_trades[
            (df_trades['instId'] == moneda) & 
            (df_trades['id'] == trade_id_state) # Comparación string vs string
        ].iloc[0]
    except IndexError:
        st.error(f"No se encontró el Trade ID '{trade_id_state}' para la moneda '{moneda}'. Por favor, verifique el ID.")
        st.stop()

    # 1. GESTIÓN VELAS
    with st.spinner(f"Sincronizando velas para {moneda}..."):
        df_velas = obtener_velas_smart(moneda, start_date, end_date)
    
    if df_velas.empty:
        st.error("No se pudieron cargar datos de velas.")
        st.stop()

    # 2. GESTIÓN OI
    symbol_oi = moneda if "-SWAP" in moneda else f"{moneda}-SWAP"
    if show_oi:
        if st.sidebar.button(f"📥 Descargar/Actualizar OI"):
            with st.spinner(f"Descargando OI para {symbol_oi}..."):
                df_nuevo_oi = descargar_oi_rango(symbol_oi, start_date, end_date)
                if not df_nuevo_oi.empty:
                    df_local_oi = cargar_datos_csv(symbol_oi, 'open_interest')
                    if not df_local_oi.empty:
                        df_final_oi = pd.concat([df_local_oi, df_nuevo_oi])
                        df_final_oi = df_final_oi.drop_duplicates(subset=['timestamp'], keep='last').sort_values('timestamp')
                    else:
                        df_final_oi = df_nuevo_oi
                    guardar_datos_csv(df_final_oi, symbol_oi, 'open_interest')
                    st.success(f"OI Actualizado.")
                    time.sleep(1)
                    st.rerun() 
        
        df_oi_disk = obtener_oi_smart(symbol_oi, start_date, end_date)
        if not df_oi_disk.empty:
            df_velas = df_velas.sort_values('timestamp')
            df_merged = pd.merge_asof(df_velas, df_oi_disk[['timestamp', 'openInterest']], on='timestamp', direction='backward', tolerance=pd.Timedelta('4h'))
            df_merged['openInterest'] = df_merged['openInterest'].ffill()
            df_velas = df_merged
        else:
            df_velas['openInterest'] = np.nan

    # 3. GENERAR GRÁFICO BASE
    fig = generate_chart(
        df_velas.copy(), trade, moneda, offset_horas,
        show_ema_9, show_ema_21, show_ema_20, show_ema_50, show_ema_200,
        show_macd, show_rsi, show_stochrsi, show_bb, show_supertrend,
        show_volume, show_oi
    )
    
    # 4. AÑADIR ETIQUETA DE PRECIO ESTÁTICA (La "Opción A")
    live_price = get_current_price(moneda)
    if live_price:
        last_close = df_velas['close'].iloc[-1]
        price_color_hex = "#26a69a" if live_price >= last_close else "#ef5350" 
        
        fig.add_hline(y=live_price, line_dash="dot", line_color=price_color_hex, line_width=2, row=1, col=1)
        fig.add_annotation(
            x=1, y=live_price, xref="paper", yref="y",
            text=f"AHORA {live_price:.4f}", showarrow=False,
            bgcolor=price_color_hex, font=dict(color="white", size=11),
            xanchor="left", xshift=2
        )

    # 5. RENDERIZAR GRÁFICO (una sola vez)
    config = {
        'scrollZoom': True, 'displayModeBar': True, 'showAxisDragHandles': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    
    with chart_placeholder.container():
        st.plotly_chart(fig, use_container_width=True, config=config)

    # 6. BUCLE DE PRECIO EN VIVO (Solo actualiza el header)
    close_price_compare = df_velas['close'].iloc[-1]
    
    while True:
        current_price_loop = get_current_price(moneda)
        
        if current_price_loop:
            color_css = "#00ff00" if current_price_loop >= close_price_compare else "#ff0000"
            
            header_placeholder.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #333;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="margin:0; color: white; font-family: sans-serif;">{moneda}</h2>
                            <span style="color: #888; font-size: 12px;">En vivo desde OKX</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 36px; color: {color_css}; font-weight: bold; font-family: monospace;">
                                {current_price_loop:,.4f}
                            </span>
                            <br>
                            <span style="font-size: 12px; color: #666;">Actualizado: {datetime.now().strftime('%H:%M:%S')}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            header_placeholder.markdown(f"### {moneda} (Precio en vivo no disponible)")

        time.sleep(1)