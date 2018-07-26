import pandas as pd
import matplotlib.pyplot as plt

filename = "data\\pokemon.csv"
df = pd.read_csv(filename)

# Pokemons por tipo
df["type1"].value_counts().plot.bar()
plt.show()

# Frecuencia de pokemons por HP
df["hp"].value_counts().sort_index().plot.line()
plt.show()

# Frecuencia de pokemons por peso
df["weight_kg"].plot.hist()
plt.show()