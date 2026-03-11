import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
data = pd.read_csv("dataset/cleaned_data.csv")

print("Dataset before encoding:")
print(data.head())

# Create LabelEncoder object
encoder = LabelEncoder()

# Encode categorical columns
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = encoder.fit_transform(data[column])

print("\nDataset after encoding:")
print(data.head())

# Save encoded dataset
data.to_csv("dataset/encoded_data.csv", index=False)

print("\nData encoding completed successfully!")
print("Encoded dataset saved as encoded_data.csv")
