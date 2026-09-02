# Predicción de incumplimiento crediticio con Random Forest

## Descripción

Este proyecto tiene como objetivo predecir si un cliente presentará incumplimiento en el pago de su tarjeta de crédito utilizando el dataset **Default of Credit Card Clients**.

Para realizar la clasificación se implementó un algoritmo **Random Forest desde cero**, sin utilizar frameworks de machine learning para la construcción del modelo.

## Dataset

El conjunto de datos contiene información demográfica, financiera y del historial de pagos de **30,000 clientes**.

La variable objetivo utilizada es:

default payment next month

* 0: No Default
* 1: Default

Durante el análisis exploratorio se identificó un desbalance entre las clases, con aproximadamente **77.88% de clientes sin incumplimiento** y **22.12% con incumplimiento**.

## Análisis exploratorio de datos

Antes de desarrollar el modelo se realizó un análisis exploratorio para conocer la estructura y características principales del dataset.

Se analizaron:

* Dimensiones y estructura del dataset.
* Valores faltantes y registros duplicados.
* Distribución de la variable objetivo.
* Variables categóricas.
* Historial de pagos.
* Distribución de variables numéricas.
* Relación entre el estado de pago reciente y el incumplimiento.
* Correlación entre las variables.

El análisis permitió identificar que las variables relacionadas con el historial de pagos presentan una relación importante con el incumplimiento de pago.

## Preprocesamiento

Antes de entrenar el modelo se realizaron los siguientes pasos:

* Eliminación de la variable ID.
* Agrupación de categorías especiales de EDUCATION.
* Recodificación de la categoría 0 de MARRIAGE.
* Separación de las variables predictoras y la variable objetivo.
* División estratificada de los datos en **70% para entrenamiento y 30% para prueba**.

## Implementación del Random Forest

La implementación incluye:

* Construcción de árboles de decisión.
* Selección aleatoria de características.
* Cálculo de entropía y ganancia de información.
* Generación de muestras mediante bootstrap.
* Combinación de las predicciones mediante voto mayoritario.

Cada árbol es entrenado utilizando una muestra aleatoria de los datos. Esto permite generar diversidad entre los árboles que forman el bosque. Finalmente, las predicciones individuales se combinan para obtener la clasificación final.

## Evaluación del modelo

El modelo se evalúa utilizando el conjunto de prueba, el cual no es utilizado durante el entrenamiento.

Debido al desbalance identificado en la variable objetivo, se utilizan diferentes métricas para evaluar el desempeño:

* Matriz de confusión.
* Accuracy.
* Precision.
* Recall.
* F1-score.

Estas métricas permiten analizar el desempeño del modelo más allá de únicamente considerar el porcentaje total de predicciones correctas.

## Cómo ejecutar el proyecto

El proyecto puede ejecutarse de dos maneras:

### Opción 1: Google Colab

Para visualizar y ejecutar el desarrollo completo del proyecto:

1. Abrir el archivo M2_AprendizajeM_sin_framework.ipynb en Google Colab.
2. Cargar el dataset default of credit card clients.xls.
3. Ejecutar las celdas del notebook en orden.

Esta opción permite reproducir el análisis exploratorio, preprocesamiento de los datos, implementación del algoritmo, entrenamiento y evaluación del modelo.

### Opción 2: Ejecutar únicamente el algoritmo

Para ejecutar el modelo sin utilizar Google Colab se pueden utilizar directamente los conjuntos de entrenamiento y prueba preprocesados incluidos en el repositorio:

* credit_default_train.csv
* credit_default_test.csv

El archivo random_forest.py carga los conjuntos de entrenamiento y prueba, entrena el Random Forest, realiza las predicciones y muestra las métricas de evaluación del modelo.

## Archivos del repositorio

* M2_AprendizajeM_sin_framework.ipynb: notebook principal con el desarrollo completo del proyecto.
* credit_default_train.csv: conjunto de datos utilizado para entrenamiento.
* credit_default_test.csv: conjunto de datos utilizado para prueba.

## Tecnologías utilizadas

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn únicamente para las métricas de evaluación
* Google Colab

## Conclusiones

En este proyecto se implementó un algoritmo Random Forest desde cero para predecir el incumplimiento de pago de clientes de tarjetas de crédito.

El desarrollo del algoritmo permitió comprender con mayor detalle los principales componentes de Random Forest, incluyendo la construcción de árboles de decisión, el uso de entropía y ganancia de información, la selección aleatoria de características, el muestreo bootstrap y la combinación de predicciones mediante voto mayoritario.

Durante el análisis de los datos se identificó un desbalance entre las clases, por lo que el desempeño del modelo no se evalúa únicamente mediante accuracy, sino también utilizando precision, recall, F1-score y la matriz de confusión.

Finalmente, la implementación desde cero presenta un mayor costo computacional que el uso de librerías optimizadas. Sin embargo, permite comprender de manera práctica el funcionamiento interno del algoritmo y cada uno de los elementos que forman un Random Forest.

## Autora

Ana Paula Moreno

