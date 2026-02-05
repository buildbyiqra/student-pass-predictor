import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("Students_Performance.csv")

print(data.head())   # check data

# ✅ Create PASS/FAIL column
data["pass"] = data["math score"].apply(lambda x: 1 if x >= 40 else 0)

# ✅ Select IMPORTANT features only
X = data[[
    "reading score",
    "writing score"
]]

y = data["pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# BEST beginner algorithm
model = RandomForestClassifier()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("MODEL ACCURACY =", accuracy)

# Save model
with open("student_pass_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("MODEL SAVED 🔥")