"""
El algoritmo OneR es bastante simple pero puede ser bastante efecivo. Es el siguiente:
    .Por cada variable
        .Por cada valor de la variable
            -La predicción basada en esta clase es la clase más frecuente
            -Procesar el error de esta predicción
        .Sumar los errores de predicción de todos los valores de la variable
    .Usar la variable con el menor error

"""
from collections import defaultdict
from operator import itemgetter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_feature_value(x, y_true, feature, value):
    # Crea un diccionario para contar con qué frecuencia se dan determinadas predicciones
    class_counts = defaultdict(int)

    # Se itera a través de cada muestra y se cuenta la frecuencia de cada par clase/valor
    for sample, y in zip(x, y_true):
        if sample[feature] == value:
            class_counts[y] += 1

    # Se obtiene la mejor ordenándola (la mayor primero) y eligiendo el primer item
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]

    # El error es el número de muestras que no se clasifican como la clase más frecuente
    # *y* tienen el valor de la característica
    error = sum([class_count for class_value, class_count in class_counts.items()
                 if class_value != most_frequent_class])
    return most_frequent_class, error


def train_on_feature(x, y_true, feature):
    """
    Computes the predictors and error for a given feature using the OneR algorithm

    Parameters
    ----------
    x: array [n_samples, n_features]
        The two dimensional array that holds the dataset. Each row is a sample, each column
        is a feature.

    y_true: array [n_samples,]
        The one dimensional array that holds the class values. Corresponds to x, such that
        y_true[i] is the class value for sample x[i].

    feature: int
        An integer corresponding to the index of the variable we wish to test.
        0 <= variable < n_features

    Returns
    -------
    predictors: dictionary of tuples: (value, prediction)
        For each item in the array, if the variable has a given value, make the given prediction.

    error: float
        The ratio of training data that this rule incorrectly predicts.
    """
    # Obtiene todos los valores únicos que tiene esta variable
    values = set(x[:, feature])
    # Almacena la matriz de predictores que se devuelve
    predictors = {}
    errors = []

    for current_value in values:
        most_frequent_class, error = train_feature_value(x, y_true, feature, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)

    # Calcula el error total de usar esta característica para clasificar
    total_error = sum(errors)
    return predictors, total_error


def predict(x_test, model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in x_test])
    return y_predicted


if __name__ == "__main__":
    # Cargamos nuestro dataset
    dataset = load_iris()
    x = dataset.data
    y = dataset.target
    n_samples, n_features = x.shape

    # Comenzamos el proceso de discretización
    # Calculamos la media de cada característica (4), siendo el primer elemento de la lista la media
    # de la primera características y así sucesivamente.
    attribute_means = x.mean(axis=0)
    # Transformamos nuestro dataset de características continuas a características categóricas discretas
    x_d = np.array(x >= attribute_means, dtype='int')

    # Para evitar el sobreajuste, y no usar el dataset de prueba para el entrenamiento,
    # dividimos el dataset en dos datasets más pequeños (aprox. un 14% del total)
    x_train, x_test, y_train, y_test = train_test_split(x_d, y, random_state=14)

    # Se procesan todos los predictores
    all_predictors = {variable: train_on_feature(x_train, y_train, variable) for variable in range(x_train.shape[1])}
    errors = {variable: error for variable, (mapping, error) in all_predictors.items()}

    # Se elige el mejor y se guarda como "model", ordenado por error
    best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
    print("The best model is based on variable {0} and has error {1:.2f}".format(best_variable, best_error))

    # Se elige el mejor model
    model = {'variable': best_variable, 'predictor': all_predictors[best_variable][0]}
    print(model)

    y_predicted = predict(x_test, model)
    print(y_predicted)

    # Se calcula la precisión tomando la media de las cantidades que cumple que y_predicted es igual a y_test
    accuracy = np.mean(y_predicted == y_test) * 100
    print("The test accuracy is {:.1f}%".format(accuracy))

    print(classification_report(y_test, y_predicted))
