import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df : pd.DataFrame, target, scale_features=False):

    #remove duplicate rows
    df = df.drop_duplicates()

    
    y = df[target]
    X = df.drop(["Addiction Level", "ProductivityLoss"], axis=1)

    # handling for missing vals
    for col in X.select_dtypes(include="number"):
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include="object"):
        X[col] = X[col].fillna(X[col].mode()[0])

    # one hot for categorical values
    X = pd.get_dummies(X, drop_first=True)

    # Scale numerical features for knn and svm
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    #train test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
