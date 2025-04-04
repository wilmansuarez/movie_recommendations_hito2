from flask import Flask, Response, render_template
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import base64
from flask import send_file

app = Flask(__name__)

# Cargar datos de películas
movies_df = pd.read_csv("data/movies.csv")

# Cargar recomendaciones precomputadas
with open("models/top_recommendations.pkl", "rb") as f:
    top_n = pickle.load(f)

@app.route("/recommend/<int:user_id>")
def recommend(user_id):
    movie_ids = top_n.get(user_id, [])
    if not movie_ids:
        return Response(f"<h2>No hay recomendaciones disponibles para el usuario {user_id}</h2>", mimetype="text/html")

    recommended = movies_df[movies_df["movieId"].isin(movie_ids)].copy()
    recommended = recommended.set_index("movieId").loc[movie_ids]
    table_html = recommended[["title", "genres"]].to_html(classes="table table-striped", index=False)
    return Response(f"<h2>Recomendaciones para el usuario {user_id}</h2>{table_html}", mimetype="text/html")

@app.route("/metrics")
def metrics():
    # Métricas comparativas entre modelos
    metrics_data = [
        {"metric": "Precisión (RMSE)", "model1": 0.85, "model2": 0.89},
        {"metric": "Costo de entrenamiento (min)", "model1": 12, "model2": 5},
        {"metric": "Costo de inferencia (ms)", "model1": 150, "model2": 80},
        {"metric": "Tamaño del modelo (MB)", "model1": 120, "model2": 75},
    ]
    return render_template('metrics.html', metrics=metrics_data)

@app.route("/metrics/online")
def online_metrics():
    # Cargar resultados Hit@10 por usuario
    df = pd.read_csv("results/hit_at_10_per_user.csv")
    avg_hit = df["hit_at_10"].mean()

    # Crear gráfico
    fig, ax = plt.subplots()
    df["hit_at_10"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Distribución de Hit@10")
    ax.set_xlabel("Hit@10 (0 = fallo, 1 = acierto)")
    ax.set_ylabel("Número de usuarios")

    # Convertir gráfico a imagen para HTML
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    html = f"""
    <h2>Evaluación Online del Modelo</h2>
    <p><strong>Hit@10 Promedio:</strong> {avg_hit:.2f}</p>
    <img src="data:image/png;base64,{plot_url}" alt="Hit@10 Distribution">
    """
    return Response(html, mimetype="text/html")

@app.route("/download/hit10")
def download_hit10():
    return send_file("results/hit_at_10_per_user.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
