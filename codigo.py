# =========================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
plt.figure(figsize=(8,5))
sns.set(style="whitegrid")

print("\n" + "="*50 + "\n")

# =========================================
# 2. CARGA DE DATOS
# =========================================
ruta = "/content/La_data_v1.xlsx"
df = pd.read_excel(ruta)

# Limpieza de nombres de columnas (importante)
df.columns = df.columns.str.strip()

print("Primeras filas:\n")
display(df.head())

print("\n" + "="*50 + "\n")

# =========================================
# 3. REVISIÓN DE TIPOS DE DATOS
# =========================================
print("Información del dataset:\n")
df.info()

print("\n" + "="*50 + "\n")

# =========================================
# 4. LIMPIEZA Y VALIDACIÓN
# =========================================

print("Valores nulos por columna:\n")
print(df.isnull().sum())

# Eliminación de duplicados (si existen)
df = df.drop_duplicates()

print("\n" + "="*50 + "\n")

# =========================================
# 5. FEATURE ENGINEERING
# =========================================

df['Price_Gap'] = df['Base Price'] - df['Total Price']

print("Nueva columna creada: Price_Gap\n")

print("\n" + "="*50 + "\n")

# =========================================
# 6. DESCRIPCIÓN ESTADÍSTICA
# =========================================
print("Estadísticas descriptivas:\n")
display(df.describe())

print("\n" + "="*50 + "\n")

# =========================================
# 7. ANÁLISIS EXPLORATORIO (EDA)
# =========================================

print("Gráficos de análisis exploratorio...\n")

# =========================================
# 7.1 GRÁFICOS DE COMPARACIÓN
# =========================================
print("GRÁFICOS DE COMPARACIÓN\n")

# Relación Precio vs Ventas
plt.figure()
sns.scatterplot(x='Total Price', y='Units Sold', data=df)
plt.title("Relación entre Precio Final y Unidades Vendidas")
plt.show()

print("\n")

# Relación Brecha de Precio vs Ventas
plt.figure()
sns.scatterplot(x='Price_Gap', y='Units Sold', data=df)
plt.title("Impacto del Descuento en las Ventas")
plt.show()

print("\n")

# =========================================
# 7.2 GRÁFICOS DE SEGMENTACIÓN
# =========================================

print("GRÁFICOS DE SEGMENTACIÓN\n")

# Ventas por tienda (usando directamente Sales Revenue del Excel)
ventas_tienda = df.groupby('Store ID')[['Units Sold', 'Sales Revenue']].sum().reset_index()

plt.figure()
sns.barplot(x='Store ID', y='Units Sold', data=ventas_tienda)
plt.title("Unidades Vendidas por Tienda")
plt.xticks(rotation=45)
plt.show()

print("\n" + "="*50 + "\n")


# =========================================
# 7.2 TOP 5 TIENDAS - GRÁFICO MEJORADO
# =========================================

print("TOP 5 TIENDAS POR UNIDADES VENDIDAS\n")

# Agrupar y ordenar
ventas_tienda = df.groupby('Store ID')[['Units Sold']].sum().reset_index()
ventas_tienda = ventas_tienda.sort_values(by='Units Sold', ascending=False)

#Agrupar el top numero 5
top_5_tiendas = ventas_tienda.head(5)

# Gráfico
plt.figure(figsize=(8,5))
ax = sns.barplot(
    x='Store ID',
    y='Units Sold',
    data=top_5_tiendas
)

# Títulos
plt.title("Top 5 Tiendas por Ventas")
plt.xlabel("Tienda")
plt.ylabel("Unidades Vendidas")

# Valores dentro de cada barra
for i, v in enumerate(top_5_tiendas['Units Sold']):
    ax.text(i, v * 0.5, f"{int(v)}",
            ha='center', va='center',
            color='white', fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*50 + "\n")



# =========================================
# 8. MATRIZ DE CORRELACIÓN (COMPARACIÓN)
# =========================================
print("Matriz de correlación:\n")

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()

print("\n" + "="*50 + "\n")




# =========================================
# 9. MACHINE LEARNING - PRONÓSTICO DE DEMANDA
# =========================================

print("=" * 50)
print("MACHINE LEARNING - PRONÓSTICO DE DEMANDA")
print("=" * 50 + "\n")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Eliminar filas con vacios
df_ml = df[['Total Price', 'Base Price', 'Price_Gap', 'Units Sold']].dropna()

# Variables para usar
X = df_ml[['Total Price', 'Base Price', 'Price_Gap']]  # variables que afectan las ventas
y = df_ml['Units Sold']                                 # lo que queremos predecir

# Dividimos los datos: 80% para enseñar, 20% para evaluar el modleo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos y entrenamos el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# El modelo predice las ventas con los datos de prueba
y_pred = modelo.predict(X_test)

# Evaluamos el modelo
r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² : {r2:.4f}  → El modelo explica el {r2*100:.1f}% de la variación en ventas")
print(f"MSE: {mse:.2f} → Error promedio del modelo\n")

# Gráfico: comparamos lo que predijo el modelo vs lo que pasó en realidad
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Predicción perfecta')
plt.title("Valores Reales vs Valores Predichos")
plt.xlabel("Unidades Reales")
plt.ylabel("Unidades Predichas")
plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "=" * 50 + "\n")

