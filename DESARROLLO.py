import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import os
#import subprocess

#Prueba
# Ruta relativa — funciona para cualquiera que clone el repo
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Leer Excel desde data inicial/ ───────────
df_prod = pd.read_excel(os.path.join(BASE, "data inicial", "BD productos.xlsx"))
df_mov  = pd.read_excel(os.path.join(BASE, "data inicial", "BD movimientos.xlsx"))
df_inv  = pd.read_excel(os.path.join(BASE, "data inicial", "BD inventario.xlsx"))

# %%
# Limpieza y estructuración de modelo productos

df_prod=df_prod.dropna(subset=["default_code","vendor_id","multiple"])
df_prod.loc[df_prod['sale_delay'] <= 10, 'sale_delay'] = 11
df_prod = df_prod[df_prod['sale_delay'] <= 365]
df_prod.loc[df_prod['multiple'] == 0, 'multiple'] = 1

# %%
# Limpieza y estructuración de modelo movimientos

df_mov=df_mov.dropna(subset=["so","customer_id"])
df_mov=df_mov.groupby(["so", "product_id"], as_index=False).agg({"mov_id":"first","wh_id":"first","customer_id":"first","product_qty": "sum","date": "max",})
df_mov["año"]=df_mov["date"].dt.year
df_mov["semana"]=df_mov["date"].dt.isocalendar().week
df_mov["semana_fecha"] = pd.to_datetime(df_mov["año"].astype(str) + "-W" + df_mov["semana"].astype(str).str.zfill(2) + "-1",format="%G-W%V-%u")

df_vta_prod=df_mov.groupby(
    ["product_id", "semana_fecha"],
    as_index=False
    ).agg(
        clientes_unicos=("customer_id", "nunique"),
        ventas=("product_qty", "sum")
    )

# %% Generación archivo rotación para html
# Filtro de últimas 24 semanas para el análisis de rotación
df_mov['semana_fecha'] = pd.to_datetime(df_mov['semana_fecha'])
fecha_corte = df_vta_prod['semana_fecha'].max() - pd.Timedelta(weeks=24)
df_vta_24 = df_vta_prod[df_vta_prod['semana_fecha'] >= fecha_corte]
df_ultima_venta = df_vta_prod.groupby("product_id", as_index=False).agg(
    ultima_venta=("semana_fecha", "max")
)

# Conteo de frecuencias y generación df rotación
df_rot=df_vta_24.groupby("product_id", as_index=False).agg(
    semanas_con_venta=("semana_fecha", "nunique"),
    ventas_totales=("ventas", "sum")
)

df_rot['avg_semanal'] = (df_rot['ventas_totales'] / 24).round(1)

def clasificar_rotacion(semanas):
    if semanas >= 11:
        return 'A'
    elif semanas >= 5:
        return 'B'
    elif semanas >= 2:
        return 'C'
    else:
        return 'D'

df_rot['cls'] = df_rot['semanas_con_venta'].apply(clasificar_rotacion)

df_inv_consolidado = df_inv.groupby("product_id", as_index=False).agg(
    on_hand=("on_hand", "sum"),
    reserved_quantity=("reserved_quantity", "sum")
)

# Merge outer para no perder nada
df_rotation = df_inv_consolidado.merge(df_rot, on="product_id", how="outer")
df_rotation = df_rotation.merge(df_ultima_venta, on="product_id", how="left")

# Llenar sin rotación como dead
df_rotation['cls'] = df_rotation['cls'].fillna('D')

# Filtrar: todos los que tienen stock + los high/medium sin stock
df_rotation = df_rotation[
    (df_rotation['on_hand'] > 0.5) |
    (df_rotation['cls'].isin(['A', 'B']))
]

# Los sin stock quedan con on_hand NaN, los llenamos con 0
df_rotation['on_hand'] = df_rotation['on_hand'].fillna(0)
df_rotation['reserved_quantity'] = df_rotation['reserved_quantity'].fillna(0)

df_rotation = df_rotation.merge(
    df_prod[['product_id', 'default_code', 'description']],
    on='product_id',
    how='left'
)

df_rotation['semanas_con_venta'] = df_rotation['semanas_con_venta'].fillna(0)
df_rotation['ultima_venta'] = df_rotation['ultima_venta'].fillna(pd.NaT)
df_rotation['default_code'] = df_rotation['default_code'].fillna('REVISAR')
df_rotation['description'] = df_rotation['description'].fillna('REVISAR')
df_rotation['avg_semanal'] = df_rotation['semanas_con_venta'].fillna(0)

df_rotation = df_rotation.rename(columns={
    'default_code': 'code',
    'description': 'desc',
    'on_hand': 'stock'
})


df_rotation.to_csv('data/rotation.csv', index=False)

# %% Generación CSV para el html del forecasting
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
import logging
logging.disable(logging.WARNING)

from tqdm import tqdm

productos_ab = df_rot[df_rot['cls'].isin(['A', 'B'])]['product_id'].unique()

df_vta_ab = df_vta_prod[df_vta_prod['product_id'].isin(productos_ab)]

# Crear calendario completo por producto
semanas = pd.date_range(
    start=df_vta_ab['semana_fecha'].min(),
    end=df_vta_ab['semana_fecha'].max(),
    freq='W-MON'
)

idx = pd.MultiIndex.from_product([productos_ab, semanas], names=['product_id', 'semana_fecha'])
df_vta_fore = pd.DataFrame(index=idx).reset_index()

# 4. Pegar ventas y rellenar 0
df_vta_fore = df_vta_fore.merge(
    df_vta_ab[['product_id', 'semana_fecha', 'ventas','clientes_unicos']],
    on=['product_id', 'semana_fecha'],
    how='left'
)
df_vta_fore['ventas'] = df_vta_fore['ventas'].fillna(0)
df_vta_fore['clientes_unicos'] = df_vta_fore['clientes_unicos'].fillna(0)

fecha_corte_fore = df_vta_fore['semana_fecha'].max() - pd.Timedelta(weeks=78)
df_vta_fore = df_vta_fore[df_vta_fore['semana_fecha'] >= fecha_corte_fore]

def crear_features(df_producto):
    df = df_producto.copy().sort_values('semana_fecha')
    
    # EWMs
    df['ewm_4']  = df['ventas'].ewm(span=4).mean().shift(1)
    df['ewm_8']  = df['ventas'].ewm(span=8).mean().shift(1)
    df['ewm_16'] = df['ventas'].ewm(span=16).mean().shift(1)
    
    # Rollings
    df['rolling_4'] = df['ventas'].shift(1).rolling(4).mean()
    df['rolling_8'] = df['ventas'].shift(1).rolling(8).mean()
    
    # Clientes únicos
    df['clientes_unicos'] = df['clientes_unicos']
    
    # Estacionalidad
    df['semana_año'] = df['semana_fecha'].dt.isocalendar().week.astype(int)
    df['mes']        = df['semana_fecha'].dt.month
    
    df = df.dropna()
    return df

def procesar_producto(pid, df_vta_fore, n_test=8, n_forecast=4):
    features = ['ewm_4', 'ewm_8', 'ewm_16', 'rolling_4', 'rolling_8', 
                'clientes_unicos', 'semana_año', 'mes']
    
    df_p = df_vta_fore[df_vta_fore['product_id'] == pid].sort_values('semana_fecha')
    serie = df_p['ventas'].values
    fechas = df_p['semana_fecha'].values
    
    if len(serie) < n_test + 10:
        return []
    
    train = serie[:-n_test]
    test  = serie[-n_test:]
    modelos = {}

    # ── 1. Promedio móvil simple ──────────────
    try:
        pred_ma = np.full(n_test, train[-4:].mean())
        modelos['MA'] = mean_squared_error(test, pred_ma)
    except:
        modelos['MA'] = np.inf

    # ── 2. EWM ───────────────────────────────
    try:
        ewm = pd.Series(train).ewm(span=8, adjust=False).mean()
        pred_ewm = np.full(n_test, ewm.iloc[-1])
        modelos['EWM'] = mean_squared_error(test, pred_ewm)
    except:
        modelos['EWM'] = np.inf

    # ── 3. Holt-Winters ───────────────────────
    try:
        hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=13).fit()
        pred_hw = hw.forecast(n_test)
        modelos['HW'] = mean_squared_error(test, pred_hw)
    except:
        modelos['HW'] = np.inf

    # ── 4. ARIMA ─────────────────────────────
    try:
        arima = auto_arima(
            train, seasonal=False, suppress_warnings=True, error_action='ignore',
            max_p=3, max_q=3, max_d=1, stepwise=True
        )
        pred_arima = arima.predict(n_periods=n_test)
        modelos['ARIMA'] = mean_squared_error(test, pred_arima)
    except:
        modelos['ARIMA'] = np.inf

    # ── 5. XGBoost ───────────────────────────
    try:
        df_feat = crear_features(df_p)
        X = df_feat[features]
        y = df_feat['ventas']
        X_train = X.iloc[:-n_test]
        y_train = y.iloc[:-n_test]
        X_test  = X.iloc[-n_test:]
        y_test  = y.iloc[-n_test:]
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        modelos['XGB'] = mean_squared_error(y_test, pred_xgb)
    except:
        modelos['XGB'] = np.inf

    # ── Mejor modelo ─────────────────────────
    mejor = min(modelos, key=modelos.get)

    # ── Forecast 4 semanas ───────────────────
    try:
        if mejor == 'MA':
            forecast = np.full(n_forecast, serie[-4:].mean())
        elif mejor == 'EWM':
            forecast = np.full(n_forecast, pd.Series(serie).ewm(span=8, adjust=False).mean().iloc[-1])
        elif mejor == 'HW':
            hw_full = ExponentialSmoothing(serie, trend='add', seasonal='add', seasonal_periods=13).fit()
            forecast = hw_full.forecast(n_forecast)
        elif mejor == 'ARIMA':
            arima_full = auto_arima(
                serie, seasonal=False, suppress_warnings=True, error_action='ignore',
                max_p=3, max_q=3, max_d=1, stepwise=True
            )
            forecast = arima_full.predict(n_periods=n_forecast)
        elif mejor == 'XGB':
            df_feat_full = crear_features(df_p)
            xgb_full = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb_full.fit(df_feat_full[features], df_feat_full['ventas'])
            last_row = df_feat_full[features].iloc[[-1]]
            forecast = []
            for _ in range(n_forecast):
                pred = xgb_full.predict(last_row)[0]
                forecast.append(pred)
    except:
        forecast = np.full(n_forecast, serie[-4:].mean())

    # ── Guardar resultados ───────────────────
    ultima_fecha = pd.Timestamp(fechas[-1])
    resultados = []
    for i, f in enumerate(forecast):
        resultados.append({
            'product_id': pid,
            'semana_fecha': ultima_fecha + pd.Timedelta(weeks=i+1),
            'forecast': max(0, round(f, 1)),
            'modelo': mejor,
            'mse': round(modelos[mejor], 2)
        })
    return resultados

from joblib import Parallel, delayed

def pipeline_forecast(df_vta_fore, n_test=8, n_forecast=4):
    productos = df_vta_fore['product_id'].unique()
    
    resultados = Parallel(n_jobs=6)(
        delayed(procesar_producto)(pid, df_vta_fore, n_test, n_forecast)
        for pid in tqdm(productos, desc='Procesando productos', unit='producto')
    )
    
    # Aplanar lista de listas
    resultados_flat = [r for sublist in resultados for r in sublist]
    return pd.DataFrame(resultados_flat)

# Correr pipeline
df_forecast = pipeline_forecast(df_vta_fore)
print(df_forecast.head())
print(f"Productos procesados: {df_forecast['product_id'].nunique()}")
#%%
# ── Calendario completo últimas 13 semanas ────
fecha_corte_hist = df_vta_fore['semana_fecha'].max() - pd.Timedelta(weeks=13)
semanas_hist = pd.date_range(
    start=fecha_corte_hist,
    end=df_vta_fore['semana_fecha'].max(),
    freq='W-MON'
)

idx = pd.MultiIndex.from_product(
    [productos_ab, semanas_hist],
    names=['product_id', 'semana_fecha']
)
df_hist_cal = pd.DataFrame(index=idx).reset_index()

# ── Pegar ventas reales ───────────────────────
df_hist_cal = df_hist_cal.merge(
    df_vta_fore[['product_id', 'semana_fecha', 'ventas']],
    on=['product_id', 'semana_fecha'],
    how='left'
)
df_hist_cal['ventas'] = df_hist_cal['ventas'].fillna(0)
df_hist_cal['tipo'] = 'historico'
df_hist_cal['modelo'] = ''
df_hist_cal = df_hist_cal.rename(columns={'ventas': 'valor'})

# ── Forecast ──────────────────────────────────
df_fore_export = df_forecast[['product_id', 'semana_fecha', 'forecast', 'modelo']].copy()
df_fore_export['tipo'] = 'forecast'
df_fore_export = df_fore_export.rename(columns={'forecast': 'valor'})

# ── Unir ambos ────────────────────────────────
df_chart = pd.concat([df_hist_cal, df_fore_export], ignore_index=True)

# ── Pegar code y desc ─────────────────────────
df_chart = df_chart.merge(
    df_prod[['product_id', 'default_code', 'description']],
    on='product_id',
    how='left'
)

df_chart = df_chart.rename(columns={
    'default_code': 'code',
    'description': 'desc'
})

# ── Exportar ──────────────────────────────────
df_chart.to_csv('data/forecast_chart.csv', index=False)

# %% Generar CSV para ver stock y los parametros de los productos esto en html pestaña de forecasting
# Parámetros de todos los productos
df_params = df_prod[['product_id', 'default_code', 'sale_delay', 'multiple']].copy()
df_params.to_csv('data/productos_params.csv', index=False)

# Stock por almacén de todos los productos
df_stock_wh = df_inv[['product_id', 'wh_id', 'on_hand', 'reserved_quantity']].copy()
df_stock_wh['almacen'] = 'Almacén ' + df_stock_wh['wh_id'].astype(str)
df_stock_wh.to_csv('data/stock_almacen.csv', index=False)

# %%
# # Solo se usa para ver el top productos con venta
# freq_producto = df_alm_prod.groupby("product_id", as_index=False).agg(
#     semanas_con_venta=("semana_fecha", "nunique"),
#     total_ventas=("ventas", "sum")
# )

# top_freq = freq_producto.sort_values(
#     "semanas_con_venta", ascending=False
# ).head(20)

# print(top_freq)

# %%
# Marcar el código y almacén a usar
# prod=10727
# alm=1

# #Ordenar la BD y filtrarla

# serie=df_alm_prod[(df_alm_prod["product_id"]==prod) & (df_alm_prod["wh_id"]==alm)]
# serie=serie.sort_values("semana_fecha")

# # Crear calendario de 2 años hacia atrás
# hoy=pd.Timestamp.today()
# inicio=(hoy-pd.DateOffset(years=2)).to_period("W").start_time
# fin=hoy.to_period("W").start_time

# calendario=pd.DataFrame({"semana_fecha":pd.date_range(start=inicio,end=fin,freq="W-MON")})

# # Unir calendario con la serie
# serie=calendario.merge(serie,on="semana_fecha",how="left")

# # Rellenar faltantes
# serie["ventas"]=serie["ventas"].fillna(0)
# serie["clientes_unicos"]=serie["clientes_unicos"].fillna(0)

# # Si quieres dejar enteros
# serie["clientes_unicos"]=serie["clientes_unicos"].astype(int)

# # Volver a poner producto y almacén
# serie["product_id"]=serie["product_id"].fillna(prod)
# serie["wh_id"]=serie["wh_id"].fillna(alm)

# # Agregar columna con los datos suavizados ES PROBABLE QUE TENGAMOS QUE METERLE SPAN 8 PARA QUE TENGA MÁS MEMORIA
# serie["ventas_suav"] = serie["ventas"].ewm(span=4).mean()

# # Graficar ventas
# plt.figure(figsize=(12,5))

# plt.plot(serie["semana_fecha"], serie["ventas"], label="Real", alpha=0.5)
# plt.plot(serie["semana_fecha"], serie["ventas_suav"], label="Exponencial")

# plt.legend()
# plt.title(f"Suavizado - Producto {prod} - WH {alm}")
# plt.xlabel("Semana")
# plt.ylabel("Ventas")

# plt.xticks(rotation=45)
# plt.show()
# # %%
# # # Generación de features para el modelo
# # LAGS (memoria reciente)
# serie["ewm_4"] = serie["ventas"].ewm(span=4).mean()
# serie["ewm_8"] = serie["ventas"].ewm(span=8).mean()
# serie["ewm_16"] = serie["ventas"].ewm(span=16).mean()
# serie["rolling_4"] = serie["ventas"].rolling(4).mean()
# serie["rolling_8"] = serie["ventas"].rolling(8).mean()
# serie["rolling_16"] = serie["ventas"].rolling(16).mean()
# serie["clientes_unicos"] = serie["clientes_unicos"]

# # #Definir variable objetivo
# serie["y"]=serie["ventas"]

# # Limpiar NA
# df_modelo = serie.dropna().copy()

# # # Definición variables
# features = [
#     "ewm_4", "ewm_8", "ewm_16",
#     "rolling_4", "rolling_8", "rolling_16",
#     "clientes_unicos"
# ]

# X=df_modelo[features]
# y=df_modelo["y"] 

# train=df_modelo.iloc[:-8]   # entrenas con todo menos últimas 8 semanas para evaluar con lo demás
# test=df_modelo.iloc[-8:]

# X_train=train[features]
# y_train=train["y"]

# X_test=test[features]
# y_test=test["y"]

# # %%
# # # Modelado random forest

# model = RandomForestRegressor(
#     n_estimators=100,
#     random_state=42
# )

# model.fit(X_train, y_train)

# pred_rf=model.predict(X_test)

# mae_rf = mean_absolute_error(y_test, pred_rf)
# media = y_test.mean()

# error_pct_rf = mae_rf / media

# print("RF Error %:", error_pct_rf)
# print("RF MAE:", mae_rf)


# # %% Modelado XGBR
# model=XGBRegressor(
#     objective="reg:squarederror",
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )

# model.fit(X_train, y_train)

# # Predicción
# pred_xgbr=model.predict(X_test)

# # Evaluación
# mae_xgbr = mean_absolute_error(y_test, pred_xgbr)
# error_pct_xgbr = mae_xgbr / y_test.mean()

# print("XGBR Error %:", error_pct_xgbr)
# print("XGBR MAE:", mae_xgbr)

# # %%
# plt.figure(figsize=(14,6))

# # Histórico (train)
# plt.plot(train["semana_fecha"], train["y"], label="Train", color="blue")
# # Real (test)
# plt.plot(test["semana_fecha"], y_test, label="Real", color="black")
# # Predicción
# plt.plot(test["semana_fecha"], pred_rf, label="RandomForest", color="red")
# plt.plot(test["semana_fecha"], pred_xgbr, label="XGBoost", color="orange", linestyle="--")
# plt.legend()
# plt.title(f"Forecast - Producto {prod} - WH {alm}")
# plt.xlabel("Semana")
# plt.ylabel("Ventas")
# plt.xticks(rotation=45)
# plt.show()
