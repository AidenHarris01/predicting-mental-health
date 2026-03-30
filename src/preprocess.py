import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df : pd.DataFrame, target, scale_features=False):
    """
    Preprocess the dataset

    Args:
        df (pd.DataFrame): Dataframe representing the dataset
        target (str): "Addiction Level" or "ProductivityLoss" [sic]
        scale_features (bool): Whether to standardize numerical features (for kNN or SVM)

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Separate features and target
    y = df[target]
    X = df.drop(["Addiction Level", "ProductivityLoss"], axis=1)

    # Handle missing values
    for col in X.select_dtypes(include="number"):
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include="object"):
        X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical values
    X = pd.get_dummies(X, drop_first=True)

    # Scale numerical features if required (kNN, SVM)
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Splits: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
