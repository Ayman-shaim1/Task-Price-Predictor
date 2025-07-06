# predict.py
import sys
import os
import joblib
from sentence_transformers import SentenceTransformer

# Lire les textes depuis les arguments de la ligne de commande
# Exemple : python3 predict.py "title" "desc" "title_task" "desc_task"
titre_prestation = sys.argv[1]
desc_prestation = sys.argv[2]
titre_tache = sys.argv[3]
desc_tache = sys.argv[4]

# Charger le modèle depuis le même dossier que ce script
model_path = os.path.join(os.path.dirname(__file__), "price_model.pkl")
embedder, model = joblib.load(model_path)

# Fusionner les textes
combined_text = " ".join([titre_prestation, desc_prestation, titre_tache, desc_tache])

# Encoder et prédire
vector = embedder.encode([combined_text])
predicted_price = max(0, model.predict(vector)[0])  # éviter les prix négatifs

# Afficher la prédiction
print(round(predicted_price, 2))
