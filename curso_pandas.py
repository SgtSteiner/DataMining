import pandas as pd
import matplotlib.pyplot as plt

"""
Este dataset contiene 150K reseñas de vinos extraidos de WineEnthusiast
"""
filename = "data\\winemag-data_first150k.csv"
df = pd.read_csv(filename, index_col=0)
print(df.columns)
print()

# Provincias con mayor nº de vinos
print(df["province"].value_counts().head(20))
df["province"].value_counts().head(20).plot.bar()
plt.show()

# Proporción de vinos por provincias
print((df["province"].value_counts() / len(df)).head(20))
(df["province"].value_counts() / len(df)).head(20).plot.bar()
plt.show()

# Puntuaciones dadas a los vinos
print(df["points"].value_counts().sort_index().head(20))
df["points"].value_counts().sort_index().head(20).plot.bar()
plt.show()
df["points"].value_counts().sort_index().head(20).plot.line()
plt.show()
df["points"].value_counts().sort_index().head(20).plot.area()
plt.show()

# Histograma. Numero de vinos con precios menores de 200$
print(df[df["price"].values < 200]["price"])
df[df["price"].values < 200]["price"].plot.hist()
plt.show()

# Histograma. Puntuaciones obtenidas por los vinos
print(df["points"])
df["points"].plot.hist()
plt.show()



