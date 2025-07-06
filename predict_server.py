from flask import Flask, request, jsonify
import joblib
from sentence_transformers import SentenceTransformer

# Charger les modèles
embedder, model = joblib.load("price_model_v2.pkl")

# Créer l'app Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Price Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    titre_prestation = data.get("titrePrestation", "")
    desc_prestation = data.get("descPrestation", "")
    titre_tache = data.get("titreTache", "")
    desc_tache = data.get("descTache", "")

    combined_text = " ".join([
        titre_prestation,
        desc_prestation,
        titre_tache,
        desc_tache
    ])

    vector = embedder.encode([combined_text])
    predicted_price = model.predict(vector)[0]

    return jsonify({"predictedPrice": round(float(predicted_price), 2)})

if __name__ == "__main__":
    app.run(port=7000)
