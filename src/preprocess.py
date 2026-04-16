import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df : pd.DataFrame, target, scale_features=False):

    #remove duplicates
    df = df.drop_duplicates()

    #separate features and target
    y = df[target]
    X = df.drop(["Addiction Level", "ProductivityLoss", "Self Control", "Satisfaction"], axis=1)

    # handle missing vals
    for col in X.select_dtypes(include="number"):
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include="object"):
        X[col] = X[col].fillna(X[col].mode()[0])

    # one hot for categorical values
    X = pd.get_dummies(X, drop_first=True)

    # train test split
    X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp)

    # scale numerical features for knn and svm
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_val, X_test, y_train, y_val, y_test