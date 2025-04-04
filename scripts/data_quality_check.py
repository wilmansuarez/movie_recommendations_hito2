import pandas as pd

ratings = pd.read_csv("data/ratings.csv")

print("Valores nulos:")
print(ratings.isnull().sum())

print("\nValores fuera de rango en 'rating':")
print(ratings[~ratings['rating'].between(0, 5)])

print("\nDistribución de ratings:")
print(ratings['rating'].value_counts().sort_index())
