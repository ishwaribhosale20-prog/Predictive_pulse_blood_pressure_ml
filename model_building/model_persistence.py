import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load training data
X_train = pd.read_csv("model_building/X_train.csv")
y_train = pd.read_csv("model_building/y_train.csv").squeeze()

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save trained model
with open("web_app/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully as model.pkl")
