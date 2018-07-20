import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from collections import defaultdict


if __name__ == "__main__":
    # Cargamos el dataset y dimensionamos los arrays X, y en función del mismo
    data_filename = "data\\ionosphere.data"
    X = np.zeros((351, 34), dtype="float")
    y = np.zeros((351, ), dtype="bool")

    with open(data_filename, "r") as input_file:
        reader = csv.reader(input_file)
        # iteramos por cada línea del dataset, donde cada línea representa un nuevo conjunto de mediciones,
        # que es una muestra en este datasets
        for i, row in enumerate(reader):
            data = [float(datum) for datum in row[:-1]]
            X[i] = data
            y[i] = row[-1] == "g"   # El último valor nos indica la clase (good, bad): True o False

    # Para evitar el sobreajuste, y no usar el dataset de prueba para el entrenamiento,
    # dividimos el dataset en dos datasets más pequeños (aprox. un 14% del total)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
    print("There are {} samples in the training dataset".format(X_train.shape[0]))
    print("There are {} samples in the testing dataset".format(X_test.shape[0]))
    print("Each sample has {} features".format(X_train.shape[1]))

    estimator = KNeighborsClassifier()
    # Ajustamos el estimador con el conjunto de entrenamiento
    estimator.fit(X_train, y_train)
    # Entrenamos y evaluamos con el conjunto de prueba
    y_predicted = estimator.predict(X_test)
    # Se calcula la precisión tomando la media de los valores que cumplen que y_predicted es igual a y_test
    accuracy = np.mean(y_test == y_predicted) * 100
    print("The accuracy is {0:.1f}%".format(accuracy))

    # Probamos la validación cross-fold
    scores = cross_val_score(estimator, X, y, scoring="accuracy")
    average_accuracy = np.mean(scores) * 100
    print("The average accuracy is {0:.1f}%".format(average_accuracy))

    # Probamos diferentes valores para el parámetro n_neighbors
    avg_scores = []
    all_scores = []
    parameters_value = list(range(1, 21))
    for n_neighbors in parameters_value:
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator, X, y, scoring="accuracy")
        avg_scores.append(np.mean(scores))
        all_scores.append(scores)

    plt.plot(parameters_value, avg_scores, "-o")
    plt.show()

    for parameter, scores in zip(parameters_value, all_scores):
        n_scores = len(scores)
        plt.plot([parameter] * n_scores, scores, "-o")
    plt.show()

    plt.plot(parameters_value, all_scores, 'bx')
    plt.show()

    # Para compensar la varianza ejecutamos 100 veces por cada parámetro de n_neighbors
    # all_scores = defaultdict(list)
    # parameter_values = list(range(1, 21))
    # for n_neighbors in parameter_values:
    #     for i in range(100):
    #         estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    #         scores = cross_val_score(estimator, X, y, scoring='accuracy', cv=10)
    #         all_scores[n_neighbors].append(scores)
    # for parameter in parameter_values:
    #     scores = all_scores[parameter]
    #     n_scores = len(scores)
    #     plt.plot([parameter] * n_scores, scores, '-o')
    # plt.show()

    # Preprocesamiento
    X_broken = np.array(X)
    X_broken[:, ::2] /= 10  # Rompemos el dataset dividiendo cada segunda característica por 10
    estimator = KNeighborsClassifier()
    original_scores = cross_val_score(estimator, X, y, scoring='accuracy')
    print("The original average accuracy for is {0: .1f} % ".format(np.mean(original_scores) * 100))
    broken_scores = cross_val_score(estimator, X_broken, y, scoring='accuracy')
    print("The 'broken' average accuracy for is {0: .1f} % ".format(np.mean(broken_scores) * 100))

    # Preprocesamiento estándar
    X_transformed = MinMaxScaler().fit_transform(X_broken)
    estimator = KNeighborsClassifier()
    transformed_scores = cross_val_score(estimator, X_transformed, y, scoring="accuracy")
    print("The transformed average accuracy for is {0: .1f} % ".format(np.mean(transformed_scores) * 100))

    # Pipelines
    scaling_pipeline = Pipeline([("scale", MinMaxScaler()),
                                 ("predict", KNeighborsClassifier())])
    scores = cross_val_score(scaling_pipeline, X_broken, y, scoring="accuracy")
    print("The pipeline scored average accuracy for is {0: .1f} % ".format(np.mean(scores) * 100))
