import os
import numpy as np
import csv

if __name__ == "__main__":
    data_filename = "data\\ionosphere.data"
    X = np.zeros((351, 34), dtype="float")
    y = np.zeros((351, ), dtype="bool")

    with open(data_filename, "r") as input_file:
        reader = csv.reader(input_file)
        for i, row in enumerate(reader):
            data = [float(datum) for datum in row[:-1]]
            X[i] = data
            y[i] = row[-1] = "g"

    print(X)