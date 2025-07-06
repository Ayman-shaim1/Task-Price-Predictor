import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer
import joblib
print("Libraries imported")

# Load the datasets
service_applications = pd.read_csv("csv/final_v4_serviceApplications.csv")
service_task_proposals = pd.read_csv("csv/final_v4_serviceTaskProposals.csv")
service_requests = pd.read_csv("csv/final_v4_serviceRequests.csv")
service_tasks = pd.read_csv("csv/final_v4_serviceTasks.csv")

#  Step 1: Filter accepted applications
accepted_apps = service_applications[service_applications["status"] == "accepted"]
accepted_ids = accepted_apps["$id"].tolist()

#  Step 2: Filter valid proposals linked to accepted applications
valid_proposals = service_task_proposals[
    service_task_proposals["serviceApplication"].isin(accepted_ids)
]

#  Step 3: Merge with serviceTasks and serviceRequests
merged = valid_proposals.merge(
    service_tasks, left_on="serviceTask", right_on="$id", suffixes=("", "_task")
).merge(
    service_requests, left_on="serviceRequest", right_on="$id", suffixes=("", "_request")
)

# Step 4: Build the training DataFrame
df = pd.DataFrame({
    "title": merged["title_request"],
    "desc": merged["description_request"],
    "title_task": merged["title"],
    "desc_task": merged["description"],
    "price": merged["newPrice"]
})

df = df.fillna("")  # Fill missing values

# 🧠 Step 5: Combine text fields into a single column
df["combined_text"] = df[["title", "desc", "title_task", "desc_task"]].agg(" ".join, axis=1)

#  Step 6: Use SentenceTransformer to encode text
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
X_vectors = model_embed.encode(df["combined_text"].tolist())
y = df["price"]

#  Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

#  Step 8: Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

#  Evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

#  Step 9: Example prediction
example = {
    "title": "Garden cleanup and tree trimming",
    "desc": "Remove fallen leaves, trim bushes, and cut overgrown tree branches from the backyard.",
    "title_task": "Clean and mow front yard",
    "desc_task": "Mow 200m² of grass in the front yard. Collect all grass and debris. Type: standard. Location: front yard. Quantity: 1"
}
example["combined_text"] = " ".join([
    example["title"],
    example["desc"],
    example["title_task"],
    example["desc_task"]
])
example_vector = model_embed.encode([example["combined_text"]])
predicted_price = model.predict(example_vector)[0]

print(predicted_price)

joblib.dump((model_embed, model), "price_model_v2.pkl")
print("Model saved")