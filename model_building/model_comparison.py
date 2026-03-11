import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load split data
X_train = pd.read_csv("model_building/X_train.csv")
X_test = pd.read_csv("model_building/X_test.csv")
y_train = pd.read_csv("model_building/y_train.csv").squeeze()
y_test = pd.read_csv("model_building/y_test.csv").squeeze()

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("Model Comparison Results:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} Accuracy: {accuracy:.2f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy, 2))
