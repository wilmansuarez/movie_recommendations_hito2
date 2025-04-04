# movie-recommender/scripts/offline_evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

def load_data():
    """Carga las calificaciones y las convierte al formato Surprise."""
    ratings = pd.read_csv("data/ratings.csv")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    return data

def evaluate_model(model, trainset, testset):
    """Entrena y evalúa un modelo en los datos de prueba."""
    model.fit(trainset)
    predictions = model.test(testset)
    rmse_score = rmse(predictions, verbose=False)
    mae_score = mae(predictions, verbose=False)
    return rmse_score, mae_score

def plot_results(results_df):
    """Genera un gráfico de barras comparando RMSE y MAE entre modelos."""
    results_df.set_index("Model", inplace=True)
    ax = results_df.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_title("Comparación de Modelos (RMSE y MAE)")
    ax.set_ylabel("Puntaje")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("reports/offline_evaluation_plot.png")
    print("Gráfico guardado en 'reports/offline_evaluation_plot.png'")

def run_offline_evaluation():
    """Ejecuta la evaluación offline con múltiples modelos."""
    data = load_data()
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    models = {
        "SVD": SVD(),
        "KNNBasic": KNNBasic(sim_options={'user_based': False})
    }

    results = []
    for model_name, model in models.items():
        rmse_score, mae_score = evaluate_model(model, trainset, testset)
        results.append({"Model": model_name, "RMSE": rmse_score, "MAE": mae_score})
        print(f"{model_name} -> RMSE: {rmse_score:.4f}, MAE: {mae_score:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/offline_evaluation_results.csv", index=False)
    print("\nResultados guardados en 'reports/offline_evaluation_results.csv'")

    plot_results(results_df)

if __name__ == "__main__":
    run_offline_evaluation()
