from flask import Flask
from joblib import load

app = Flask(__name__)
model = load('models/best_model.pkl')  # puede ser Popularity o CF

@app.route('/recommend/<int:user_id>')
def recommend(user_id):
    recs = model.recommend(user_id)
    return ','.join(map(str, recs[:20]))
