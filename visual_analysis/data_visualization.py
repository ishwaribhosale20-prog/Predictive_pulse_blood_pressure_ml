import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/data.txt")

print("Dataset Preview:")
print(data.head())

# Set style
sns.set(style="whitegrid")

# Age vs Blood Pressure
plt.figure()
sns.scatterplot(x=data["age"], y=data["systolic"])
plt.title("Age vs Systolic Blood Pressure")
plt.xlabel("Age")
plt.ylabel("Systolic BP")
plt.show()

# BMI distribution
plt.figure()
sns.histplot(data["bmi"], kde=True)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.show()

# Heart Rate vs Diastolic BP
plt.figure()
sns.scatterplot(x=data["heart_rate"], y=data["diastolic"])
plt.title("Heart Rate vs Diastolic BP")
plt.xlabel("Heart Rate")
plt.ylabel("Diastolic BP")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
