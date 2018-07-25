from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def export_tree(filename, decision_tree, feature_names=None, class_names=None):
    with open(filename, "w") as f_out:
        f_out = export_graphviz(decision_tree,
                                out_file=f_out,
                                max_depth=7,
                                impurity=True,
                                feature_names=feature_names,
                                class_names=class_names,
                                rounded=True,
                                filled=True,)
    return f_out


if __name__ == "__main__":
    # Creamos el dataset con los resultados de los partidos de la NBA
    data_filename = "data\\basketball.csv"
    dataset = pd.read_csv(data_filename, parse_dates=["Date"])
    # Completamos las columnas del dataset
    dataset.columns = ["Date", "Start (ET)", "Visitor Team", "VisitorPTS",
                       "Home Team", "HomePTS", "OT?", "Score Type", "Attend.", "Notes"]

    # Añadimos una nueva característica para indicar si ganó el equipo local y creamos el dataset con la clase (y)
    dataset["HomeWin"] = dataset["HomePTS"] > dataset["VisitorPTS"]
    y_true = dataset["HomeWin"].values

    print("El porcentaje de veces que gana el equipo local es {0:.2f}%".format(dataset["HomeWin"].mean() * 100))

    print(dataset.groupby("HomeWin").size())
    sns.catplot('HomeWin', data=dataset, kind="count")
    plt.show()

    # Añadimos dos nuevas características para indicar si el equipo local y el visitante ganaron su último partido
    won_last = defaultdict(int)
    dataset["HomeLastWin"] = 0
    dataset["VisitorLastWin"] = 0

    for index, row in dataset.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        dataset.set_value(index, "HomeLastWin", won_last[home_team])
        dataset.set_value(index, "VisitorLastWin", won_last[visitor_team])

        won_last[home_team] = int(row["HomeWin"])
        won_last[visitor_team] = 1 - int(row["HomeWin"])

    print(dataset.head(6))
    print(dataset.loc[1000:1005])

    # Creamos el árbol de decisión
    clf = DecisionTreeClassifier(random_state=14)
    X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]]

    scores = cross_val_score(clf, X_previouswins, y_true, scoring="accuracy")
    print("Accuracy X_previouswin: {0:.1f}%".format(np.mean(scores) * 100))

    # Dibujamos el árbol
    clf = clf.fit(X_previouswins, y_true)
    fout = export_tree("tree1.dot",
                       clf,
                       feature_names=X_previouswins.columns,
                       class_names=["Lost", "Won"])

    # Creamos el dataset de clasificaciones de la NBA
    standings_filename = "data\\standings.csv"
    standings = pd.read_csv(standings_filename, skiprows=1)
    standings.head()

    # Añadimos una nueva característica para indicar la clasificación de los equipos
    dataset["HomeTeamRanksHigher"] = 0
    for index, row in dataset.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        home_rank = standings[standings["Team"] == home_team]["Rk"].values[0]
        visitor_rank = standings[standings["Team"] == visitor_team]["Rk"].values[0]
        dataset.set_value(index, "HomeTeamRanksHigher", int(home_rank < visitor_rank))

    X_homehigher = dataset[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
    clf = DecisionTreeClassifier(random_state=14)

    scores = cross_val_score(clf, X_homehigher, y_true, scoring="accuracy")
    print("Accuracy X_homehiguer: {0:.1f}%".format(np.mean(scores) * 100))

    # Añadimos una nueva característica para indicar el resultado del último enfrentamiento entre ambos equipos
    last_match_winner = defaultdict(int)
    dataset["HomeTeamWonLast"] = 0

    for index, row in dataset.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        teams = tuple(sorted([home_team, visitor_team]))    # sort para un ordenamiento consistente
        # Establecemos en la  fila quién ganó el último encuentro
        home_team_won_last = 1 if last_match_winner[teams] == row["Home Team"] else 0
        dataset.set_value(index, "HomeTeamWonLast", home_team_won_last)
        # ¿Quién ganó éste?
        winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
        last_match_winner[teams] = winner

    X_lastwinner = dataset[["HomeTeamWonLast", "HomeTeamRanksHigher", "HomeLastWin", "VisitorLastWin"]].values
    cls = DecisionTreeClassifier(random_state=14, criterion="entropy")

    scores = cross_val_score(clf, X_lastwinner, y_true, scoring="accuracy")
    print("Accuracy X_lastwinner: {0:.1f}%".format(np.mean(scores) * 100))

    # Transformamos a características categóricas con LabelEncoder
    encoding = LabelEncoder()
    encoding.fit(dataset["Home Team"].values)
    home_teams = encoding.transform(dataset["Home Team"].values)
    visitor_teams = encoding.transform(dataset["Visitor Team"].values)
    X_teams = np.vstack([home_teams, visitor_teams]).T

    onehot = OneHotEncoder()
    X_teams = onehot.fit_transform(X_teams).todense()
    clf = DecisionTreeClassifier(random_state=14)

    scores = cross_val_score(clf, X_teams, y_true, scoring="accuracy")
    print("Accuracy X_teams: {0:.1f}%".format(np.mean(scores) * 100))

    # Aplicando random forests
    clf = RandomForestClassifier(random_state=14)
    scores = cross_val_score(clf, X_teams, y_true, scoring="accuracy")
    print("Accuracy X_teams random forests: {0:.1f}%".format(np.mean(scores) * 100))

    # Añadimos más características para comprobar su rendimiento
    X_all = np.hstack([X_lastwinner, X_teams])
    clf = RandomForestClassifier(random_state=14)
    scores = cross_val_score(clf, X_all, y_true, scoring="accuracy")
    print("Accuracy X_teams random forests: {0:.1f}%".format(np.mean(scores) * 100))

    # Uso de GridSearchCV
    parameter_space = {
        "max_features": [2, 10, "auto"],
        "n_estimators": [100, 200],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [2, 4, 6],
    }
    clf = RandomForestClassifier(random_state=14)
    grid = GridSearchCV(clf, parameter_space)
    grid.fit(X_all, y_true)
    print("Accuracy GridSearchCV random forests: {0:.1f}%".format(grid.best_score_ * 100))
    print(grid.best_estimator_)
