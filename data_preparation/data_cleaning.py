import pandas as pd

# Load dataset
data = pd.read_csv("dataset/data.txt")

# Display first rows
print("Original Dataset:")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values with mean
data = data.fillna(data.mean())

# Remove duplicate rows
data = data.drop_duplicates()

# Convert column names to lowercase
data.columns = data.columns.str.lower()

# Save cleaned dataset
data.to_csv("dataset/cleaned_data.csv", index=False)

print("\nData cleaning completed successfully!")
print("Cleaned dataset saved as cleaned_data.csv")
