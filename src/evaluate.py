import sys
import pandas as pd
import pickle

from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess

# Get command-line argument for target (Either "Addiction Level" or "ProductivityLoss" [sic])
if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = "Addiction Level"
print(f"Training models for target: {target}")

# Load dataset
df = pd.read_csv("data/Time_Wasters_on_Social_Media.csv")

# Preprocess
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df, target, scale_features=True)

# Load trained models
with open(f"{target}_models.pkl", "rb") as f:
    models = pickle.load(f)

for name, model in models.items():
    print(f"\nEvaluating {name}:")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
