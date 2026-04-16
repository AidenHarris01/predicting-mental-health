import os
import sys
import pandas as pd
import pickle
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from preprocess import preprocess
from models import get_models

# cli for target
if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    target = "Addiction Level"
print(f"Training models for target: {target}")


data_path = os.path.join(os.path.dirname(__file__), "..", "data", "Time_Wasters_on_Social_Media.csv")
df = pd.read_csv(data_path)
X_train, X_val, X_test,y_train, y_val,y_test = preprocess(df, target, scale_features=True)

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

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
split_indices = np.array([-1] * len(X_train) + [0] * len(X_val))
ps = PredefinedSplit(split_indices)

best_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    grid = GridSearchCV(model, param_grids.get(name, {}),cv=ps, scoring="f1_macro")
    grid.fit(X_trainval, y_trainval)
    best_models[name] = grid.best_estimator_
    print(f"Best params for {name}: {grid.best_params_}")

#save 
output_path = os.path.join(os.path.dirname(__file__), "..", f"{target}_models.pkl")
with open(output_path, "wb") as f:
    pickle.dump(best_models, f)