import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load training and testing data
X_train = pd.read_csv("model_building/X_train.csv")
X_test = pd.read_csv("model_building/X_test.csv")
y_train = pd.read_csv("model_building/y_train.csv").squeeze()
y_test = pd.read_csv("model_building/y_test.csv").squeeze()

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed")
print("Accuracy:", round(accuracy, 2))
