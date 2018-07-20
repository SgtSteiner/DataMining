from collections import defaultdict
import pandas as pd


if __name__ == "__main__":
    data_filename = "data\\basketball.csv"
    dataset = pd.read_csv(data_filename, parse_dates=["Date"])
    # Completamos las columnas del dataset
    dataset.columns = ["Date", "Start (ET)", "Visitor Team", "VisitorPTS",
                       "Home Team", "HomePTS", "OT?", "Score Type", "Attend.", "Notes"]

    # Añadimos una nueva característica para indicar si ganó el equipo local
    dataset["HomeWin"] = dataset["HomePTS"] > dataset["VisitorPTS"]
    y_true = dataset["HomeWin"].values

    print("El porcentaje de veces que gana el equipo local es {0:.2f}%".format(dataset["HomeWin"].mean() * 100))

    # Añadimos dos nuevas características para indicar si el equipo local y el visitante ganaron su último partido
    won_last = defaultdict(int)
    dataset["HomeLastWin"] = 0
    dataset["VisitorLastWin"] = 0

    for index, row in dataset.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        # row["HomeLastWin"] = won_last[home_team]
        dataset.set_value(index, "HomeLastWin", won_last[home_team])
        dataset.set_value(index, "VisitorLastWin", won_last[visitor_team])

        won_last[home_team] = int(row["HomeWin"])
        won_last[visitor_team] = 1 - int(row["HomeWin"])

    print(dataset.head(6))
    print(dataset.loc[1000:1005])

