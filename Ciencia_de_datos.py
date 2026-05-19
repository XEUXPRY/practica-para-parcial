# importamos utilidades
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import matplotlib
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import folium
import webbrowser
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colors as mcolors
from branca.colormap import linear

# 1. Cargar datos
df = pd.read_csv(r"C:\Users\Usuario\Downloads\practica para parcial\rentcrime.csv")

# 2. Vista general
print("🔹 Shape:", df.shape)
print("\n🔹 Info:")
print(df.info())

print("\n🔹 Primeras filas:")
print(df.head())

# 3. Tipos de variables
print("\n🔹 Tipos de variables:")
print(df.dtypes)

# 4. Valores nulos (clave)
print("\n🔹 Valores nulos por columna:")
print(df.isnull().sum())

print("\n🔹 Porcentaje de nulos:")
null_pct = df.isnull().mean() * 100
print(null_pct.sort_values(ascending=False))

# 5. Estadísticas básicas (solo numéricas)
print("\n🔹 Estadísticas descriptivas:")

# Opción A: Transpuesto (más legible en terminal)
print(df.describe().T)

# Opción B: Configurar pandas para ancho ilimitado (solo una vez al inicio)
# pd.set_option('display.width', None)
# pd.set_option('display.max_columns', None)
# print(df.describe())

# 6. Columnas categóricas
cat_cols = df.select_dtypes(include=['object']).columns
print("\n🔹 Variables categóricas:")
print(cat_cols)

# 7. Cardinalidad (útil para encoding después)
for col in cat_cols:
    print(f"\n🔹 {col} - valores únicos: {df[col].nunique()}")
    print(df[col].value_counts().head())
# Con este script ya sabes:

#✔ Tamaño del dataset
#✔ Tipos de datos
#✔ Dónde hay nulos
#✔ Cuánto afectan (%)
#✔ Distribución básica
#✔ Variables categóricas
#✔ Cardinalidad (clave para encoding)

####################################desde aca vamos aplicar encoding y tambien histograma para ver la distribucion de la datta##############################
# Copia del dataframe
df_model = df.copy()

# 🔹 1. amenities (binaria)
df_model['amenities'] = df_model['amenities'].map({
    'basic': 0,
    'luxury': 1
})

# 🔹 2. has_photo (categórica → one-hot)
df_model = pd.get_dummies(df_model, columns=['has_photo'], drop_first=True)

# 🔹 3. source (categórica → one-hot)
df_model = pd.get_dummies(df_model, columns=['source'], drop_first=True)

################################# histograma con todos los datos #################################


plt.figure(figsize=(8,5))
sns.histplot(df_model['price'], kde=True)
plt.title("Distribución de Price")
plt.xlabel("Price")
plt.ylabel("Frecuencia")
plt.show()

############################# histograma con todos los datos pero USANDO LOG #################################


plt.figure(figsize=(8,5))
sns.histplot(np.log(df_model['price']), kde=True)
plt.title("Distribución de Log(Price)")
plt.xlabel("Log(Price)")
plt.ylabel("Frecuencia")
plt.show()

############################# Matrix de correlacion ###############################################################

# Selecciona solo las columnas numéricas para la correlación
columnas_numericas = df_model.select_dtypes(include=['number']).columns
# Seleccionar top N variables más correlacionadas con 'price'
variable_objetivo = 'price'
n_top = 8  # Número de variables a mostrar (sin incluir la objetivo)

correlaciones = df_model[columnas_numericas].corr()[variable_objetivo].sort_values(ascending=False)

# Seleccionar top variables (excluyendo la propia variable objetivo)
top_vars = correlaciones[1:n_top+1].index.tolist() + correlaciones[-n_top:].index.tolist()
top_vars = list(set(top_vars))  # Eliminar duplicados
top_vars.append(variable_objetivo)  # Agregar la variable objetivo

# Crear heatmap solo con estas variables
plt.figure(figsize=(10, 8))
sns.heatmap(df_model[top_vars].corr(), 
            annot=True, 
            fmt='.2f',
            cmap='coolwarm', 
            center=0,
            square=True)
plt.title(f'Top correlaciones con {variable_objetivo}', fontsize=14)
plt.tight_layout()
plt.show()

############################# Definimos variables dependientes e independientes ###############################################################
variables = ['square_feet', 'bedrooms', 'bathrooms', 'medIncome']
X = df_model[variables]
y = np.log(df_model['price'])  # usamos log

X = sm.add_constant(X)

############################# Modelo OLS ###############################################################
model_ols = sm.OLS(y, X).fit()
print(model_ols.summary())

############################# Vemos si en el modelo hay temas de multicolinealidad ###############################################################


vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

############################# Llamamos al modelo GAMMA para compararlo con el OLS ###############################################################
model_gamma = sm.GLM(
    df_model['price'],
    X,
    family=sm.families.Gamma(link=sm.families.links.log())
).fit()

print(model_gamma.summary())

################################ Llamamos al modelo GWR #############################################################

#matplotlib.use('Agg')
# coordenadas
#coords = df_model[['longitude', 'latitude']].values

# variables independientes
#X_gwr = df_model[variables].values

# variable dependiente (usa log también)
#y_gwr = np.log(df_model['price']).values.reshape((-1,1))

#bw = Sel_BW(coords, y_gwr, X_gwr).search()

#gwr_model = GWR(coords, y_gwr, X_gwr, bw).fit()
#print(gwr_model.summary())

############################################### Viendo el comportamiento de los modelos realizamos TEST DE HETEROCEDASTICIDAD##################################
residuals = model_ols.resid
exog = model_ols.model.exog

bp_test = het_breuschpagan(residuals, exog)

labels = ['LM Stat', 'LM p-value', 'F Stat', 'F p-value']
print(dict(zip(labels, bp_test)))

#Graficamos
plt.scatter(model_ols.fittedvalues, residuals)
plt.axhline(0, color='red')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

############################Viendo que el modelo tiene heterocedasticidad pasamos a mirar el tema de los outliers con COOK'S como primero##############
influence = model_ols.get_influence()
cooks = influence.cooks_distance[0]

threshold = 4 / len(df_model)

outliers = np.where(cooks > threshold)[0]
print(len(outliers))

leverage = influence.hat_matrix_diag
############################################Generamos KMEANS ###########################
############################################
# 🔹 KMEANS
############################################
coords = df_model[['latitude', 'longitude']]

kmeans = KMeans(n_clusters=12,  random_state=42)
df_model['zone'] = kmeans.fit_predict(coords)

############################################
# 🔹 DUMMIES
############################################
df_model = pd.get_dummies(df_model, columns=['zone'], drop_first=True)

############################################
# 🔹 VARIABLES
############################################
zone_cols = [col for col in df_model.columns if col.startswith('zone_')]

variables_geo = [
    'square_feet',
    'bedrooms',
    'bathrooms',
    'medIncome'
] + zone_cols

############################################
# 🔹 MATRICES
############################################
X_geo = df_model[variables_geo].copy()
y = np.log(df_model['price'])

# 🔥 FORZAR A NUMÉRICO (CLAVE)
X_geo = X_geo.astype(float)

############################################
# 🔹 MODELO
############################################
X_geo = sm.add_constant(X_geo)

model_geo = sm.OLS(y, X_geo).fit(cov_type='HC3')
print(model_geo.summary())

##############################################Entrenamos el modelo para ver si hay sobre ajustes cuando se aumentan # KMEANS ###########################
# split primero
X = df_model[['square_feet','bedrooms','bathrooms','medIncome','latitude','longitude']]
y = np.log(df_model['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KMeans SOLO en train
kmeans = KMeans(n_clusters=12, random_state=42)
X_train['zone'] = kmeans.fit_predict(X_train[['latitude','longitude']])

# aplicar al test
X_test['zone'] = kmeans.predict(X_test[['latitude','longitude']])

# dummies
X_train = pd.get_dummies(X_train, columns=['zone'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['zone'], drop_first=True)

# alinear columnas
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# quitar lat/lon si quieres solo zonas
X_train = X_train.drop(columns=['latitude','longitude'])
X_test = X_test.drop(columns=['latitude','longitude'])

# modelo
X_train = sm.add_constant(X_train).astype(float)
X_test = sm.add_constant(X_test).astype(float)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

print("R2 test:", r2_score(y_test, y_pred))

############################### encontramos el numero de kmeans perfecto ahora vamos a validar supuestos#################################
##diagnostico final sobre nuestro modelo para terminar de validar temas de heterocedasticidad, su forma del error y si el modelo esta bien
# predicciones en test
y_pred = model.predict(X_test)

# residuos
residuals = y_test - y_pred

# gráfico
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)

plt.axhline(0)
plt.xlabel("Predicted (log price)")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")

plt.show()

####vemos la importancia de las variables finales de nuestro modelo
# coeficientes
coefs = model.params.drop('const')

# ordenar por impacto absoluto
importance = pd.DataFrame({
    'variable': coefs.index,
    'coef': coefs.values,
    'abs_coef': np.abs(coefs.values)
}).sort_values(by='abs_coef', ascending=False)

print(importance.head(15))

#### para ver el top de variables y su coeficientes en el modelo
plt.figure(figsize=(10,6))
sns.barplot(data=importance.head(15), x='abs_coef', y='variable')

plt.title("Top Variables Importance (abs coef)")
plt.xlabel("Absolute Coefficient")
plt.ylabel("Variable")

plt.show()

### por ultimo una visualizacion del cluster
df_model['zone'] = kmeans.predict(df_model[['latitude','longitude']])
plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df_model['longitude'],
    y=df_model['latitude'],
    hue=df_model['zone'],  # usa la original antes de dummies si la tienes
    palette='tab10',
    alpha=0.5
)

plt.title("Spatial Clusters (KMeans Zones)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.legend(title="Zone", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

### por ultimo coloreamos los precios por zona
plt.figure(figsize=(8,6))

plt.scatter(
    df_model['longitude'],
    df_model['latitude'],
    c=df_model['price'],
    cmap='viridis',
    alpha=0.5
)

plt.colorbar(label='Price')

plt.title("Price Distribution by Location")

plt.show()
###########Generamos un mapa donde ejecutivos puedan interpretarlo mejo
# centro del mapa
# centro del mapa
center_lat = df_model['latitude'].mean()
center_lon = df_model['longitude'].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

# reconstruir zone si no está
df_model['zone'] = kmeans.predict(df_model[['latitude','longitude']])

# colormap moderno
colormap = matplotlib.colormaps.get_cmap('tab20')
n_clusters = df_model['zone'].nunique()

def get_color(cluster):
    return colors.rgb2hex(colormap(cluster / n_clusters))

# sample
sample = df_model.sample(5000, random_state=42)

# loop (SOLO agregar puntos)
for _, row in sample.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2,
        color=get_color(row['zone']),
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# 🔥 guardar UNA VEZ
m.save("mapa_clusters.html")

# 🔥 abrir
webbrowser.open("mapa_clusters.html")

##################mapa pero por precio
# mapa base
m2 = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB positron")

# 🔥 usar log(price)
prices = np.log(sample['price'])

# 🔥 recorte de outliers
p_low, p_high = np.percentile(prices, [1, 99])
prices_clipped = np.clip(prices, p_low, p_high)

# normalización
norm = mcolors.Normalize(vmin=prices_clipped.min(), vmax=prices_clipped.max())

# 🔥 colores turbo (visual real)
cmap = matplotlib.colormaps.get_cmap('turbo')

# 🔥 leyenda (compatible SIEMPRE)
colormap = linear.viridis.scale(prices_clipped.min(), prices_clipped.max())
colormap.caption = "Log(Price)"
colormap.add_to(m2)

# loop
for _, row in sample.iterrows():
    value = np.log(row['price'])
    value = np.clip(value, p_low, p_high)

    color = mcolors.to_hex(cmap(norm(value)))

    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        fill=True,
        fill_color=color,
        color=None,
        fill_opacity=0.85,
        tooltip=f"Price: ${int(row['price'])}"
    ).add_to(m2)

# guardar
m2.save("mapa_precios.html")

# abrir
webbrowser.open("mapa_precios.html")