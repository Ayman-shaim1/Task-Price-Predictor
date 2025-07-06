# predict.py
import sys
import joblib
import os
from sentence_transformers import SentenceTransformer

# Récupérer les entrées depuis Node.js
titre_prestation = sys.argv[1]
desc_prestation = sys.argv[2]
titre_tache = sys.argv[3]
desc_tache = sys.argv[4]

# Charger le modèle
embedder, model = joblib.load("price_model_v2.pkl")

# Fusionner le texte
combined_text = " ".join([titre_prestation, desc_prestation, titre_tache, desc_tache])

# Transformer le texte
vector = embedder.encode([combined_text])
predicted_price = model.predict(vector)[0]

print(round(predicted_price, 2))
