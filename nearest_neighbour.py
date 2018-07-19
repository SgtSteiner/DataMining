import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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
