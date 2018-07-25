import pandas as pd


if __name__ == "__main__":
    # Creamos el dataset con el rating de las películas
    ratings_filename = "data\\ml-100k\\u.data"
    all_ratings = pd.read_csv(ratings_filename,
                              delimiter="\t",
                              header=None,
                              names=["UserID", "MovieID", "Rating", "Datetime"])

    # Convertimos a fecha la columna Datetime, dado que viene en formato unix segundos desde 1/1/1970 UTC
    all_ratings["Datetime"] = pd.to_datetime(all_ratings["Datetime"], unit="s")
    all_ratings.head()
