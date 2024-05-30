import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import requests
import time
from streamlit_echarts import st_echarts
import plotly.express as px


st.header("Simulación Monte Carlo")
st.subheader("Santiago García López")

st.subheader("¿Qué es una simulación Monte Carlo?")
st.write("Una simulación Monte Carlo es una técnica que utiliza números aleatorios para resolver problemas complejos o hacer predicciones en situaciones donde no se conoce el resultado exacto. Se basa en realizar un gran número de experimentos o simulaciones para obtener una aproximación cercana al resultado real. Es como lanzar una moneda muchas veces para estimar la probabilidad de que salga cara o cruz.")

st.markdown("---")

# Cargar los datos
df = pd.read_csv('https://research-watchlists.s3.amazonaws.com/df_UniversidadPanamericana_ohlc.csv')
df['time'] = pd.to_datetime(df['time'])

# Calcular la volatilidad implícita
def calculate_volatility(df):
    vol_df = pd.DataFrame()
    for symbol in df['Symbol'].unique():
        _df = df[df['Symbol'] == symbol]
        log_returns = np.log(_df['close'] / _df['close'].shift(1))
        vol = log_returns.std() * np.sqrt(252)  # Annualize the volatility
        vol_df = vol_df.append({'Symbol': symbol, 'Volatility': vol}, ignore_index=True)
    return vol_df

vol_df = calculate_volatility(df)

# Función para generar gráficos con streamlit-echarts
def generate_echarts_line(data, x_col, y_col, title):
    options = {
        "title": {
            "text": title
        },
        "tooltip": {
            "trigger": 'axis'
        },
        "xAxis": {
            "type": 'category',
            "data": data[x_col].tolist()
        },
        "yAxis": {
            "type": 'value'
        },
        "series": [{
            "data": data[y_col].tolist(),
            "type": 'line'
        }]
    }
    st_echarts(options=options)

    # Mostrar tabla de volatilidades implícitas
    st.subheader("Tabla de Volatilidades Implícitas")
    st.dataframe(vol_df)

# Creación de pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Ejemplo Práctico de predicción de una utilidad bruta", 
                                  "Simulador de Precios de Acciones",
                                  "Calcular el precio de un Call (Black-Scholes-Merton)",
                                  "Calcular el precio de un Call (Euler Discretization)",
                                    "Datos Financieros de finviz",
                                    "Volatilidad Implicita"])

# Ejercicio 1: Predicción de una utilidad bruta
with tab1:
    st.subheader("Ejemplo Práctico de predicción de una utilidad bruta:")
    def simulate_business(iterations=1000, rev_m=170, rev_stdev=20, cogs_mean=0.6, cogs_stdev=0.1):
        rev = np.random.normal(rev_m, rev_stdev, iterations)
        cogs = -(rev * np.random.normal(cogs_mean, cogs_stdev))
        return rev, cogs

    iterations = st.slider('Número de iteraciones', min_value=100, max_value=5000, value=1000, step=100)
    rev_m = st.slider('Ingresos medios', min_value=100, max_value=300, value=170, step=10)
    rev_stdev = st.slider('Desviación estándar de ingresos', min_value=5, max_value=50, value=20, step=5)

    rev, cogs = simulate_business(iterations, rev_m, rev_stdev)
    Gross_Profit = rev + cogs

    st.subheader('Gráfico de Ingresos')
    fig_rev = plt.figure(figsize=(10, 6))
    plt.plot(rev)
    st.pyplot(fig_rev)

    st.subheader('Gráfico de Costo de Bienes Vendidos (COGS)')
    fig_cogs = plt.figure(figsize=(10, 6))
    plt.plot(cogs)
    st.pyplot(fig_cogs)

    st.subheader('Histograma de Beneficio Bruto (bins=9)')
    fig_gross_profit = plt.figure(figsize=(10, 6))
    plt.hist(Gross_Profit, bins=[30, 40, 50, 60, 70, 80, 90, 110, 120])
    st.pyplot(fig_gross_profit)

    st.subheader('Histograma de Beneficio Bruto (bins=20)')
    fig_gross_profit_20 = plt.figure(figsize=(10, 6))
    plt.hist(Gross_Profit, bins=20)
    st.pyplot(fig_gross_profit_20)

# Ejercicio 2: Simulador de Precios de Acciones
with tab2:
    st.header('Simulador de Precios de Acciones')
    st.write("Ingrese el símbolo de la acción del S&P 500:")

    ticker = st.text_input('Símbolo de la acción:', 'AAPL')

    def simulate_stock(ticker, t_intervals=1000, iterations=10):
        data = pd.DataFrame()
        data[ticker] = yf.download(ticker, start='2014-01-01')['Adj Close']
        log_returns = np.log(1 + data.pct_change())
        u = log_returns.mean().values
        var = log_returns.var().values
        drift = u - (0.5 * var)
        stdev = log_returns.std().values

        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))

        S0 = data.iloc[-1].values
        price_list = np.zeros_like(daily_returns)
        price_list[0] = S0

        for t in range(1, t_intervals):
            price_list[t] = price_list[t - 1] * daily_returns[t]

        return price_list

    sim_button = st.button('Simular')

    if sim_button:
        st.write(f"Simulación para {ticker}:")
        price_list = simulate_stock(ticker)
        plt.figure(figsize=(10,6))
        plt.plot(price_list)
        plt.title(f"Simulación de precios para {ticker}")
        plt.xlabel("Días")
        plt.ylabel("Precio de Cierre")
        st.pyplot()

# Ejercicio 3: Calcular el precio de un Call mediante el Black-Scholes-Merton
with tab3:
    st.header("Calcular el precio de un Call mediante el Black-Scholes-Merton")

    def d1(S, K, r, stdev, T):
        return (np.log(S / K) + (r + stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))

    def d2(S, K, r, stdev, T):
        return (np.log(S / K) + (r - stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))

    def BSM(S, K, r, stdev, T):
        return (S * norm.cdf(d1(S, K, r, stdev, T))) - (K * np.exp(-r * T) * norm.cdf(d2(S, K, r, stdev, T)))

    def simulate_stock_bsm(ticker, K, T, r=0.025, t_intervals=1000, iterations=10):
        data = pd.DataFrame()
        data[ticker] = yf.download(ticker, start='2014-01-01')['Adj Close']
        log_returns = np.log(1 + data.pct_change())
        u = log_returns.mean().values
        var = log_returns.var().values
        drift = u - (0.5 * var)
        stdev = log_returns.std().values

        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))

        S0 = data.iloc[-1].values
        price_list = np.zeros_like(daily_returns)
        price_list[0] = S0
        call_price = BSM(S0, K, r, stdev, T)

        for t in range(1, t_intervals):
            price_list[t] = price_list[t - 1] * daily_returns[t]
        return price_list, call_price

    ticker = st.text_input('Símbolo de la acción:', 'AAPL', key='bsm')

    K = st.number_input('Precio de Ejercicio (K)', min_value=0.0, step=1.0, value=110.0)
    T = st.number_input('Tiempo de Vencimiento (T)', min_value=0, step=1, value=1)

    calc_button = st.button('Calcular')

    if calc_button:
        st.write(f"Simulación para {ticker}:")
        price_list, call_price = simulate_stock_bsm(ticker, K, T)

        plt.figure(figsize=(10,6))
        plt.plot(price_list)
        plt.title(f"Simulación de precios para {ticker}")
        plt.xlabel("Días")
        plt.ylabel("Precio de Cierre")
        st.pyplot()

        st.write(f"Precio del call con K={K} y T={T}: {call_price}")

# Ejercicio 4: Calcular el precio de un Call mediante Euler Discretization
with tab4:
    st.header("Calcular el precio de un Call mediante Euler Discretization")
    
    def simulate_stock_euler(ticker, r, T, t_intervals=250, iterations=10000):
        data = pd.DataFrame()
        data[ticker] = yf.download(ticker, start='2014-01-01')['Adj Close']
        log_returns = np.log(1 + data.pct_change())
        stdev = log_returns.std() * 250 ** 0.5
        stdev = stdev.values

        delta_t = T / t_intervals

        Z = np.random.standard_normal((t_intervals + 1, iterations))
        S = np.zeros_like(Z)
        S0 = data.iloc[-1].values
        S[0] = S0

        for t in range(1, t_intervals + 1):
            S[t] = S[t-1] * np.exp((r - 0.5 * stdev ** 2) * delta_t + stdev * np.sqrt(delta_t) * Z[t])

        p = np.maximum(S[-1] - 110, 0)
        C = np.exp(-r * T) * np.sum(p) / iterations

        return S, C

    st.write("Ingrese el símbolo de la acción del S&P 500:")
    rticker = st.text_input('Símbolo de la acción:', 'AAPL', key='euler')

    r = st.number_input('Tasa de Interés (r)', min_value=0.0, step=0.01, value=0.025)
    T = st.number_input('Tiempo de Vencimiento (T) en años', min_value=0.0, step=0.5, value=1.0)
    s_button = st.button('Otorgar')

    if s_button:
        st.write(f"Simulación para {rticker}:")
        S, call_price = simulate_stock_euler(rticker, r, T)

        plt.figure(figsize=(10,6))
        plt.plot(S[:, :10])
        plt.title(f"Simulación de precios para {rticker}")
        plt.xlabel("Pasos de Tiempo")
        plt.ylabel("Precio de Cierre")
        st.pyplot()

        st.write(f"Precio del call con K=110 y T={T} años: {call_price}")

# Datos Financieros de Finviz
with tab5:
    st.header("Datos Financieros de Finviz")
# Configuración inicial
    FinViz_Structure = {
        'Overview': '111',
        'Valuation': '121',
        'Financial': '161',
        'Ownership': '131',
        'Performance': '141',
        'Technical': '171'
    }

    sectores_disponibles = {
        'Any': '',
        'Basic Materials': 'sec_basicmaterials',
        'Communication Services': 'sec_communicationservices',
        'Consumer Cyclical': 'sec_consumercyclical',
        'Consumer Defensive': 'sec_consumerdefensive',
        'Energy': 'sec_energy',
        'Financial': 'sec_financial',
        'Healthcare': 'sec_healthcare',
        'Industrials': 'sec_industrials',
        'Real Estate': 'sec_realestate',
        'Technology': 'sec_technology',
        'Utilities': 'sec_utilities'
    }

    token = "656f0298-9515-406a-b19a-d683cf4deaf9"
    End_Point_1 = "https://elite.finviz.com/export.ashx?v="

    # Guardar el último sector descargado
    last_downloaded_sector = st.session_state.get('last_downloaded_sector', '')

    # Selector de sector
    st.markdown("### Select the sector and category")
    selected_sector = st.selectbox("Sector", list(sectores_disponibles.keys()))

    # Selección de categoría y visualización de datos
    selected_category = st.selectbox("Category", list(FinViz_Structure.keys()))

    # Descarga de datos
    if st.button("Descargar Data"):
        st.write("Perame...:")

        downloaded_successfully = True
        sector_filter = sectores_disponibles[selected_sector] if selected_sector != 'Any' else ''

        for key, value in FinViz_Structure.items():
            url = f"{End_Point_1}{value}&f=cap_largeover|cap_midunder,exch_nyse|nasd,idx_sp500,{sector_filter},sh_opt_option&auth={token}"
            response = requests.get(url)
            if response.status_code == 200:
                filename = f"{key}.csv"
                with open(filename, "wb") as file:
                    file.write(response.content)
            else:
                downloaded_successfully = False

            time.sleep(2)  # Pausa para evitar limitaciones de la API

        if downloaded_successfully:
            st.session_state['last_downloaded_sector'] = selected_sector
            st.success(f"The data of the {selected_sector} sector has been downloaded successfully")
            
    # Mostrar datos si el sector coincide
    if selected_category and (selected_sector == st.session_state.get('last_downloaded_sector', '')):
        filename = f"{selected_category}.csv"
        data = pd.read_csv(filename, index_col='No.')
        st.write(data)

        # Preparar datos para el MapTree
        def create_treemap_data(df):
            data = []
            for index, row in df.iterrows():
                data.append({
                    'name': row['Ticker'],
                    'value': row['Market Cap'] if 'Market Cap' in row else 0,
                    'children': [
                        {'name': 'Price', 'value': row['Price'] if 'Price' in row else 0},
                        {'name': 'P/E', 'value': row['P/E'] if 'P/E' in row else 0},
                        {'name': 'Dividend', 'value': row['Dividend'] if 'Dividend' in row else 0},
                    ]
                })
            return data

        if not data.empty:
            treemap_data = create_treemap_data(data)
            options = {
                "series": [
                    {
                        "type": 'treemap',
                        "data": treemap_data,
                        "label": {
                            "show": True,
                            "formatter": '{b}: {@value}'
                        }
                    }
                ]
            }

            st_echarts(options=options, height="500px")
# Pestaña de Volatilidad Implícita
with tab6:
    st.header("Volatilidad Implícita")

    # Selección de tickers para las gráficas
    st.subheader("Selecciona los Tickers para las Gráficas")
    selected_tickers = st.multiselect("Selecciona los Tickers", vol_df['Symbol'].unique(), default=vol_df['Symbol'].unique())

    # Filtrar datos basados en los tickers seleccionados
    filtered_vol_df = vol_df[vol_df['Symbol'].isin(selected_tickers)]

    # Gráfico de barras con Matplotlib
    st.subheader("Gráfico de Barras de Volatilidades Implícitas")
    fig_vol = plt.figure(figsize=(10, 6))
    plt.bar(filtered_vol_df['Symbol'], filtered_vol_df['Volatility'])
    plt.title('Volatilidades Implícitas por Símbolo')
    plt.xlabel('Símbolo')
    plt.ylabel('Volatilidad Implícita')
    st.pyplot(fig_vol)

    # Gráfico de líneas con streamlit-echarts
    st.subheader("Gráfico de Líneas de Volatilidades Implícitas")
    generate_echarts_line(filtered_vol_df, 'Symbol', 'Volatility', 'Volatilidades Implícitas por Símbolo')

    # Gráfico de dispersión con Matplotlib
    st.subheader("Gráfico de Dispersión de Volatilidades Implícitas")
    fig_scatter = plt.figure(figsize=(10, 6))
    plt.scatter(filtered_vol_df['Symbol'], filtered_vol_df['Volatility'], c='r', marker='o')
    plt.title('Volatilidades Implícitas por Símbolo')
    plt.xlabel('Símbolo')
    plt.ylabel('Volatilidad Implícita')
    st.pyplot(fig_scatter)

    # Calcular y mostrar estadísticas de la volatilidad implícita
    st.subheader("Estadísticas de la Volatilidad Implícita")
    mean_vol = filtered_vol_df['Volatility'].mean()
    std_vol = filtered_vol_df['Volatility'].std()
    st.write(f"Volatilidad Implícita Media: {mean_vol:.2f}")
    st.write(f"Desviación Estándar de la Volatilidad Implícita: {std_vol:.2f}")

    # Gráfico de histograma de la volatilidad implícita
    st.subheader("Distribución de la Volatilidad Implícita")
    fig_hist = plt.figure(figsize=(10, 6))
    plt.hist(filtered_vol_df['Volatility'], bins=20, edgecolor='black')
    plt.title('Distribución de la Volatilidad Implícita')
    plt.xlabel('Volatilidad Implícita')
    plt.ylabel('Frecuencia')
    st.pyplot(fig_hist)

    # Cargar datos
    df = pd.read_csv('https://research-watchlists.s3.amazonaws.com/df_UniversidadPanamericana_ohlc.csv')
    df['time'] = pd.to_datetime(df['time'])

    # Crear una lista de símbolos únicos
    rtickers = df['Symbol'].unique()

    # Filtrar datos basados en los tickers seleccionados
    filtered_vol_df = vol_df[vol_df['Symbol'].isin(selected_tickers)]

    # Cargar datos
    df = pd.read_csv('https://research-watchlists.s3.amazonaws.com/df_UniversidadPanamericana_ohlc.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    # Selector de tickers
    selected_tickers = st.multiselect('Seleccione los tickers:', tickers, default=tickers[:5])

    # Filtrar datos basados en los tickers seleccionados
    filtered_vol_df = df[df['Symbol'].isin(selected_tickers)]

    # Mostrar tabla de volatilidades implícitas
    st.subheader("Tabla de Volatilidades Implícitas")
    st.write(filtered_vol_df)

    # Gráfico 3D de la volatilidad implícita
    st.subheader("Gráfico 3D de Volatilidades Implícitas")
    fig_3d = px.scatter_3d(
        filtered_vol_df,
        x='time',
        y='Symbol',
        z='impVolatility',  # Corregido a 'impVolatility'
        color='Symbol',
        title='Volatilidades Implícitas en 3D'
    )
    st.plotly_chart(fig_3d)


    
st.set_option('deprecation.showPyplotGlobalUse', False)
