import pandas as pd
from sklearn.model_selection import train_test_split

# Load encoded dataset
data = pd.read_csv("dataset/data.txt")

print("Dataset Preview:")
print(data.head())

# Features and target
X = data[["age", "systolic", "diastolic", "bmi", "heart_rate"]]
y = data["stage"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Save split datasets
X_train.to_csv("model_building/X_train.csv", index=False)
X_test.to_csv("model_building/X_test.csv", index=False)
y_train.to_csv("model_building/y_train.csv", index=False)
y_test.to_csv("model_building/y_test.csv", index=False)

print("\nData splitting completed successfully!")
print("Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
