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

Durante el análisis exploratorio se identificó un desbalance entre las clases, con aproximadamente 77.88% de clientes sin incumplimiento y 22.12% con incumplimiento.

## Preprocesamiento

Antes de entrenar el modelo se realizaron los siguientes pasos:

* Eliminación de la variable ID.
* Agrupación de categorías especiales de EDUCATION.
* Recodificación de la categoría 0 de MARRIAGE.
* Separación de las variables predictoras y la variable objetivo.
* División estratificada de los datos en 70% para entrenamiento y 30% para prueba.

No se aplicó normalización o estandarización, ya que los árboles de decisión realizan divisiones mediante umbrales y no requieren que las variables se encuentren en la misma escala.

## Implementación del modelo

El Random Forest fue implementado sin utilizar frameworks de machine learning.

La implementación incluye:

* Construcción de árboles de decisión.
* Selección aleatoria de características.
* Cálculo de entropía y ganancia de información.
* Generación de muestras mediante bootstrap.
* Combinación de las predicciones mediante voto mayoritario.

## Evaluación

El desempeño del modelo se evalúa utilizando:

* Matriz de confusión
* Accuracy
* Precision
* Recall
* F1-score

Estas métricas permiten evaluar el comportamiento del modelo considerando el desbalance existente entre las clases.

## Archivos

* M2_AprendizajeM_sin_framework.ipynb: notebook con el análisis, preprocesamiento, implementación y evaluación del modelo.
* credit_default_train.csv: conjunto de entrenamiento.
* credit_default_test.csv: conjunto de prueba.

## Tecnologías utilizadas

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn únicamente para las métricas de evaluación
* Google Colab

## Autora

Ana Paula Moreno
