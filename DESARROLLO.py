import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

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

df_alm_prod=df_mov.groupby(
    ["product_id", "wh_id", "semana_fecha"],
    as_index=False
    ).agg(
        clientes_unicos=("customer_id", "nunique"),
        ventas=("product_qty", "sum")
    )

# %%
# Solo se usa para ver el top productos con venta
freq_producto = df_alm_prod.groupby("product_id", as_index=False).agg(
    semanas_con_venta=("semana_fecha", "nunique"),
    total_ventas=("ventas", "sum")
)

top_freq = freq_producto.sort_values(
    "semanas_con_venta", ascending=False
).head(20)

print(top_freq)

# %%
# Marcar el código y almacén a usar
prod=10727
alm=1

#Ordenar la BD y filtrarla

serie=df_alm_prod[(df_alm_prod["product_id"]==prod) & (df_alm_prod["wh_id"]==alm)]
serie=serie.sort_values("semana_fecha")

# Crear calendario de 2 años hacia atrás
hoy=pd.Timestamp.today()
inicio=(hoy-pd.DateOffset(years=2)).to_period("W").start_time
fin=hoy.to_period("W").start_time

calendario=pd.DataFrame({"semana_fecha":pd.date_range(start=inicio,end=fin,freq="W-MON")})

# Unir calendario con la serie
serie=calendario.merge(serie,on="semana_fecha",how="left")

# Rellenar faltantes
serie["ventas"]=serie["ventas"].fillna(0)
serie["clientes_unicos"]=serie["clientes_unicos"].fillna(0)

# Si quieres dejar enteros
serie["clientes_unicos"]=serie["clientes_unicos"].astype(int)

# Volver a poner producto y almacén
serie["product_id"]=serie["product_id"].fillna(prod)
serie["wh_id"]=serie["wh_id"].fillna(alm)

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
# %%
# # Generación de features para el modelo
# LAGS (memoria reciente)
serie["ewm_4"] = serie["ventas"].ewm(span=4).mean()
serie["ewm_8"] = serie["ventas"].ewm(span=8).mean()
serie["ewm_16"] = serie["ventas"].ewm(span=16).mean()
serie["rolling_4"] = serie["ventas"].rolling(4).mean()
serie["rolling_8"] = serie["ventas"].rolling(8).mean()
serie["rolling_16"] = serie["ventas"].rolling(16).mean()
serie["clientes_unicos"] = serie["clientes_unicos"]

# #Definir variable objetivo
serie["y"]=serie["ventas"]

# Limpiar NA
df_modelo = serie.dropna().copy()

# # Definición variables
features = [
    "ewm_4", "ewm_8", "ewm_16",
    "rolling_4", "rolling_8", "rolling_16",
    "clientes_unicos"
]

X=df_modelo[features]
y=df_modelo["y"] 

train=df_modelo.iloc[:-8]   # entrenas con todo menos últimas 8 semanas para evaluar con lo demás
test=df_modelo.iloc[-8:]

X_train=train[features]
y_train=train["y"]

X_test=test[features]
y_test=test["y"]

# %%
# # Modelado random forest

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred_rf=model.predict(X_test)

mae_rf = mean_absolute_error(y_test, pred_rf)
media = y_test.mean()

error_pct_rf = mae_rf / media

print("RF Error %:", error_pct_rf)
print("RF MAE:", mae_rf)


# %% Modelado XGBR
model=XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Predicción
pred_xgbr=model.predict(X_test)

# Evaluación
mae_xgbr = mean_absolute_error(y_test, pred_xgbr)
error_pct_xgbr = mae_xgbr / y_test.mean()

print("XGBR Error %:", error_pct_xgbr)
print("XGBR MAE:", mae_xgbr)

# %%
plt.figure(figsize=(14,6))

# Histórico (train)
plt.plot(train["semana_fecha"], train["y"], label="Train", color="blue")
# Real (test)
plt.plot(test["semana_fecha"], y_test, label="Real", color="black")
# Predicción
plt.plot(test["semana_fecha"], pred_rf, label="RandomForest", color="red")
plt.plot(test["semana_fecha"], pred_xgbr, label="XGBoost", color="orange", linestyle="--")
plt.legend()
plt.title(f"Forecast - Producto {prod} - WH {alm}")
plt.xlabel("Semana")
plt.ylabel("Ventas")
plt.xticks(rotation=45)
plt.show()