# 1. IMPORTAR LIBRERÍAS

# Sirve para trabajar con datos, gráficos y machine learning
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



# 2. CARGAR DATASET

# Sirve para leer el archivo donde están los datos
df = pd.read_excel("La_data_v1.xlsx")


# 3. LIMPIEZA DE DATOS

# Sirve para ver si hay datos vacíos (NaN)
print("Valores nulos por columna:")
print(df.isna().sum())

# Sirve para eliminar filas con datos faltantes
df = df.dropna()



# 4. CREAR VARIABLE DESCUENTO

# Sirve para medir cuánto descuento se aplicó
df["Price_Gap"] = df["Base Price"] - df["Total Price"]



# 5. DEFINIR VARIABLES

# X = variables que afectan las ventas
X = df[["Total Price", "Base Price", "Price_Gap"]]

# y = lo que queremos predecir (ventas)
y = df["Units Sold"]


# 6. DIVIDIR DATOS

# Sirve para separar datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# 7. ENTRENAR MODELO

# Sirve para crear el modelo de regresión lineal
modelo = LinearRegression()

# Sirve para enseñarle al modelo usando los datos
modelo.fit(X_train, y_train)



# 8. HACER PREDICCIONES

# Sirve para que el modelo prediga ventas
y_pred = modelo.predict(X_test)



# 9. EVALUACIÓN

# R2 mide qué tan bueno es el modelo (entre 0 y 1)
r2 = r2_score(y_test, y_pred)

# MSE mide el error del modelo
mse = mean_squared_error(y_test, y_pred)

print("\nRESULTADOS DEL MODELO:")
print("R2:", r2)
print("Error cuadrático medio:", mse)


# 10. COEFICIENTES

# Sirve para ver cómo influye cada variable en las ventas
coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": modelo.coef_
})

print("\nCoeficientes del modelo:")
print(coeficientes)



# 11. GRÁFICO REAL vs PREDICHO
# Valores reales
plt.figure()
plt.plot(y_test.values)
plt.title("Valores reales")
plt.xlabel("Observaciones")
plt.ylabel("Ventas")
plt.show()
# Valores predichos
plt.figure()
plt.plot(y_pred)
plt.title("Valores predichos")
plt.xlabel("Observaciones")
plt.ylabel("Ventas")
plt.show()
# Sirve para comparar valores reales vs los predichos por el modelo

plt.scatter(y_test, y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Real vs Predicho")
plt.show()