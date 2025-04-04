import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle

# Rutas relativas
DATA_DIR = "../data/"
MODELS_DIR = "../models/"
os.makedirs(MODELS_DIR, exist_ok=True)

# Cargar ratings
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

# ------------------ MODELO 1: POPULARIDAD ------------------
# Calcular promedio de ratings por película
popular_movies = ratings.groupby("movieId")["rating"].mean().sort_values(ascending=False)
popular_movies = popular_movies.head(20)  # top 20

# Guardar modelo como pickle
with open(os.path.join(MODELS_DIR, "popularity_model.pkl"), "wb") as f:
    pickle.dump(popular_movies, f)

print("Modelo de popularidad entrenado y guardado.")

# ------------------ MODELO 2: FILTRADO COLABORATIVO ------------------
# Usar Surprise con algoritmo SVD
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

algo = SVD()
algo.fit(trainset)

# Guardar modelo entrenado
with open(os.path.join(MODELS_DIR, "svd_model.pkl"), "wb") as f:
    pickle.dump(algo, f)

print("Modelo de filtrado colaborativo (SVD) entrenado y guardado.")
