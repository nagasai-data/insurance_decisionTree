# 📦 Importing all the libraries I need for this project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📂 Load the data
data = pd.read_csv("train.csv")

# 🧼 Quick overview of shape and missing values
print("🧾 Shape of the dataset:", data.shape)
print("\n🔍 Missing values in each column:\n", data.isnull().sum())

# 🔄 Convert categorical values to numbers so the model can understand
data["Gender"] = data["Gender"].map({'Male': 1, 'Female': 0})
data["Vehicle_Damage"] = data["Vehicle_Damage"].map({'Yes': 1, 'No': 0})
data["Vehicle_Age"] = data["Vehicle_Age"].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})

# 🧹 Dropping the ID column because it's not useful for prediction
data = data.drop(columns=["id"])

# 🧠 Define features and target
X = data.drop(columns=["Response"])  # Everything except target
y = data["Response"]                 # Target column

# 🧪 Splitting the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Training a Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 🔮 Make predictions
y_pred = model.predict(X_test)

# ✅ Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy on Test Set: {accuracy:.4f}")

# 📋 Detailed classification report
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# 📊 Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
