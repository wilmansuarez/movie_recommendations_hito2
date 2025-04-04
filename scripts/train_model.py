import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle
from collections import defaultdict

# Cargar ratings.csv
ratings = pd.read_csv("data/ratings.csv")

# Definir formato para Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

# Separar datos
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Entrenar modelo
model = SVD()
model.fit(trainset)

# Guardar modelo
with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Obtener lista completa de usuarios y películas
all_movie_ids = ratings["movieId"].unique()
all_user_ids = ratings["userId"].unique()

# Generar top-20 recomendaciones por usuario
def get_top_n(predictions, n=20):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for (iid, _) in user_ratings[:n]]
    return top_n

# Predecir para cada usuario todas las películas no vistas
predictions = []
for uid in all_user_ids:
    seen = ratings[ratings.userId == uid]["movieId"].values
    unseen = [iid for iid in all_movie_ids if iid not in seen]
    for iid in unseen:
        predictions.append(model.predict(uid, iid))

top_n = get_top_n(predictions, n=20)

# Guardar recomendaciones
with open("models/top_recommendations.pkl", "wb") as f:
    pickle.dump(top_n, f)
