import numpy as np
from collections import defaultdict
from operator import itemgetter


def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Regla: Si una persona compra {0} también comprará {1}".format(premise_name, conclusion_name))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))


if __name__ == "__main__":

    # Nombre de las features, por referencia
    features = ["pan", "leche", "queso", "manzanas", "plátanos"]

    # Cargamos el dataset con las muestras de compras realizadas de las diferentes features
    dataset_filename = "datasets\\affinity_dataset.txt"
    x = np.loadtxt(dataset_filename)
    n_samples, n_features = x.shape
    print("Este dataset contiene {0} muestras y {1} features".format(n_samples, n_features))

    valid_rules = defaultdict(int)
    invalid_rules = defaultdict(int)
    num_occurances = defaultdict(int)

    # Se procesan todas las reglas posibles
    for sample in x:
        for premise in range(len(features)):
            if sample[premise] == 0:
                continue
            # Se registra que la premisa fue comprada en la transacción
            num_occurances[premise] += 1
            for conclusion in range(len(features)):
                if premise == conclusion:   # se excluyen casos como "si se compra pan también se compra pan"
                    continue
                if sample[conclusion] == 1:
                    # Está persona también compró la conclusión
                    valid_rules[(premise, conclusion)] += 1
                else:
                    # Esta persona compró la premisa, pero no la conclusión
                    invalid_rules[(premise, conclusion)] += 1

    support = valid_rules
    confidence = defaultdict(float)
    for premise, conclusion in valid_rules.keys():
        rule = (premise, conclusion)
        confidence[rule] = valid_rules[rule] / num_occurances[premise]

    sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)
    for index in range(5):
        print("\nRegla nº{}".format(index + 1))
        premise, conclusion = sorted_support[index][0]
        print_rule(premise, conclusion, support, confidence, features)

    sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
    for index in range(5):
        print("\nRegla nº{}".format(index + 1))
        premise, conclusion = sorted_confidence[index][0]
        print_rule(premise, conclusion, support, confidence, features)