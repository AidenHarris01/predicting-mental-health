import sys
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV
from preprocess import preprocess
from models import get_models

# cli for target
if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = "Addiction Level"
print(f"Training models for target: {target}")

# Load dataset
df = pd.read_csv("data/Time_Wasters_on_Social_Media.csv")

# Preprocess
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df, target, scale_features=True)

# Load models
models = get_models()


param_grids = {
    "KNN": {"n_neighbors": [3,5,7]},
    "DecisionTree": {"max_depth": [5,10,None]},
    "RandomForest": {"n_estimators": [50,100], "max_depth": [5,10,None]},
    "GradientBoosting": {"n_estimators": [50,100], "learning_rate": [0.01,0.1]},
    "LogisticRegression": {"C": [0.1,1,10]},
    "SVM": {"C": [0.1,1,10], "kernel": ["linear","rbf"]},
    "NaiveBayes": {}
}

best_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    grid = GridSearchCV(model, param_grids.get(name, {}),cv=3, scoring="f1_macro")
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"best params for {name}: {grid.best_params_}")

# Save models
with open(f"{target}_models.pkl", "wb") as f:
    pickle.dump(best_models, f)
