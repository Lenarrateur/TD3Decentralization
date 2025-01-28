# train.py
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Initialize models and balances
models = {
    "random_forest": RandomForestClassifier(random_state=42),
    #"svm": SVC(probability=True, random_state=42),
    #"logistic_regression": LogisticRegression(random_state=42)
}

balances = {
    "random_forest": 1000,
    #"svm": 1000,
    #"logistic_regression": 1000
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate models
def evaluate_models():
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
    return results

# Consensus prediction
def consensus_prediction(features):
    predictions = {}
    for name, model in models.items():
        pred = model.predict_proba([features])[0]
        predictions[name] = pred

    # Average predictions for consensus
    avg_prediction = np.mean(
        [predictions[model] for model in predictions], axis=0
    )
    return avg_prediction


# Save models and balances
train_state = {
    "models": models,
    "balances": balances
}

def save_state():
    with open("train_state.json", "w") as f:
        json.dump({"balances": balances}, f)

def load_state():
    global balances
    with open("train_state.json", "r") as f:
        state = json.load(f)
        balances = state["balances"]

# Standardized API response format
def standard_response(model_name, prediction, probabilities):
    return {
        "model": model_name,
        "prediction": int(prediction),
        "probabilities": probabilities.tolist(),
        "balance": balances[model_name]
    }

# Save state after training
save_state()
