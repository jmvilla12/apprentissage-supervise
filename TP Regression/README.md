# Supervised Learning: Linear Regression

## Introduction

- Antes de empezar, lee bien las transparencias del curso.  
- Objetivos:  
  1. Entender la regresión lineal usando las fórmulas del curso.  
  2. Usar librerías de Python para aplicar regresión a conjuntos de datos simulados y reales. En particular usaremos `numpy`, `pandas` y `scikit-learn`.  
- Sé curioso: juega con los parámetros (p. ej. número de datos de entrenamiento, varianza del ruido, …), cambia la función subyacente y trata de entender los efectos.  
- Evita usar ChatGPT u otros LLMs para resolver el TP. ¡Estás aquí para cometer errores, razonar y aprender!  
- Implementa todos los ejercicios del TP en uno o varios notebooks de Python. Usa abundantes comentarios y celdas Markdown para explicar tu código.

---

## 1. Linear regression

1. **Función**  
   Crear `simple_linear_regression(X, y)` que reciba  
   - un vector de variables descriptoras `X`  
   - un vector de respuestas `y`  
   e implemente la regresión lineal simple:  
   $$
     \bar x = \frac{1}{N}\sum_{i=1}^N x_i,\quad
     \bar y = \frac{1}{N}\sum_{i=1}^N y_i,
     \quad
     \hat\beta_1 = \frac{\sum_{i=1}^N (y_i - \bar y)(x_i - \bar x)}{\sum_{i=1}^N (x_i - \bar x)^2},
     \quad
     \hat\beta_0 = \bar y - \hat\beta_1\,\bar x.
   $$

2. **Caso práctico**  
   Un hipermercado tiene 20 cajas. Estudiamos el tiempo medio de espera (en minutos) `y` en función del número de cajas disponibles `x`.  
   Dataset de tamaño \(N=7\):

   | x (cajas) | 3  | 4   | 5   | 6 | 8 | 10  | 12  |
   |-----------|----|-----|-----|---|---|-----|-----|
   | y (min)   | 16 | 12  | 9.5 | 8 | 6 | 4.5 | 4   |

   El objetivo es ajustar un modelo lineal que prediga el tiempo medio de espera dado el número de cajas.

   a) **Visualización de datos**  
   - Convertir `x` y `y` en arreglos de `numpy` y dibujar un scatter plot:  
     ```python
     import matplotlib.pyplot as plt
     plt.scatter(X, y)
     ```

   b) **Estadísticas**  
   - Estimar medias, varianzas, covarianza y coeficiente de correlación usando `numpy.mean`, `numpy.var`, `numpy.cov`, `numpy.corrcoef`.  
   - ¿Están correladas `x` e `y`?

   c) **Entrenamiento**  
   - Llamar a `simple_linear_regression(x, y)` para obtener \(\hat\beta_0\) y \(\hat\beta_1\).  
   - Dibujar la línea de regresión sobre el scatter plot.

   d) **Test / Predicción**  
   - Predecir el tiempo medio de espera para 1, 7 y 20 cajas.  
   - ¿Es apropiado el modelo lineal para estos datos?

---

## 2. Polynomial regression

1. **Generación de datos artificiales**  
   Crear una función que genere \(N=80\) muestras \((x_i, y_i)\) con  
   \[
     x_i \sim U(0,1),\quad
     \varepsilon_i \sim \mathcal{N}(0,4),\quad
     y_i = 10 + 5\,x_i + 4\sin(10\,x_i) + \varepsilon_i.
   \]

2. **Regresión lineal multidimensional**  
   - Definir `linear_regression(X, y)` que calcule  
     \(\displaystyle \hat\beta = (X^T X)^{-1} X^T y.\)  
   - La matriz \(X\) debe incluir la columna de 1s para el intercepto:
     \[
       X = \begin{bmatrix}
         1 & x_{1,1} & x_{1,2} & \dots & x_{1,d}\\
         1 & x_{2,1} & x_{2,2} & \dots & x_{2,d}\\
         \vdots & \vdots & \vdots & \ddots & \vdots\\
         1 & x_{N,1} & x_{N,2} & \dots & x_{N,d}
       \end{bmatrix}.
     \]
   - Pista: el pseudo-inverso Moore–Penrose se obtiene con `numpy.linalg.pinv(X)` y la multiplicación de matrices en Python usa `@`.

3. **Base de funciones polinómicas**  
   - Crear `phi(X, order)` que transforme cada valor \(x\) en el vector \([1, x, x^2, \dots, x^\text{order}]\).

4. **Experimento**  
   - Ajustar `linear_regression(X, y)` al dataset del ejercicio 1a.  
   - Ajustar `linear_regression(phi(X, 20), y)` al mismo dataset.  
   - Dibujar ambas curvas junto al scatter plot.  
   - ¿Cuál modelo sobreajusta y cuál infraajusta?

---

## 3. Regression con scikit-learn

### Módulos y clases útiles

| Módulo                              | Función / Clase                           |
|-------------------------------------|-------------------------------------------|
| `sklearn.linear_model`              | `LinearRegression`, `Ridge`, `Lasso`      |
| `sklearn.metrics`                   | `mean_squared_error`, `r2_score`          |
| `sklearn.model_selection`           | `KFold`, `train_test_split`               |
| `sklearn.datasets`                  | `load_diabetes`                           |
| `sklearn.feature_selection`         | `SequentialFeatureSelector`               |

---

### 3.1 Simple linear regression

1. **Lectura y visualización**  
   - Con `pandas.read_csv()` importa `house_rent.csv` (545 pisos de París: renta vs superficie).  
   - Dibuja un scatter plot. ¿Crees que un modelo lineal es adecuado?  
   - Filtra outliers (p. ej. renta < 10000).

2. **Partición de datos**  
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)