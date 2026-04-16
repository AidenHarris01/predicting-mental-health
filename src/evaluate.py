import sys
import pandas as pd
import pickle

from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess

#eval script

# cli for target
if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = "Addiction Level"
print(f"Training models for target: {target}")

df = pd.read_csv("data/Time_Wasters_on_Social_Media.csv")
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df, target, scale_features=True)


with open(f"{target}_models.pkl", "rb") as f:
    models = pickle.load(f)

for name, model in models.items():
    print(f"\nEvaluating {name}:")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
