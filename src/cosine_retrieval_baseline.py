import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import Normalizer

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess


K_CANDIDATES = [1, 3, 5, 7, 9, 11, 15, 21, 31]


def cosine_retrieval_predict(X_ref, y_ref, X_query, k):
    """Predict labels with top-k cosine-similar neighbors and majority vote."""
    X_ref_arr = np.asarray(X_ref)
    X_query_arr = np.asarray(X_query)

    # After standardization, L2-normalize vectors so the dot product equals cosine similarity.
    normalizer = Normalizer(norm="l2")
    X_ref_norm = normalizer.fit_transform(X_ref_arr)
    X_query_norm = normalizer.transform(X_query_arr)

    similarities = X_query_norm @ X_ref_norm.T
    y_ref_arr = np.asarray(y_ref)
    predictions = []

    for row in similarities:
        top_k_idx = np.argsort(row)[::-1][:k]
        neighbor_labels = y_ref_arr[top_k_idx]
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        predictions.append(labels[np.argmax(counts)])

    return np.array(predictions)


def evaluate_target(df, target):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(
        df, target, scale_features=True
    )

    best_k = None
    best_val_f1 = None
    best_val_acc = None

    for k in K_CANDIDATES:
        val_pred = cosine_retrieval_predict(X_train, y_train, X_val, k=k)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")

        if (
            best_val_f1 is None
            or val_f1 > best_val_f1
            or (val_f1 == best_val_f1 and val_acc > best_val_acc)
        ):
            best_k = k
            best_val_f1 = val_f1
            best_val_acc = val_acc

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    test_pred = cosine_retrieval_predict(X_trainval, y_trainval, X_test, k=best_k)

    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")

    return {
        "target": target,
        "best_k": best_k,
        "val_accuracy": best_val_acc,
        "val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    }


def main():
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "Time_Wasters_on_Social_Media.csv"
    )
    df = pd.read_csv(data_path)

    for target in ["Addiction Level", "ProductivityLoss"]:
        result = evaluate_target(df, target)
        print(f"\nTarget: {result['target']}")
        print(f"Best k on validation: {result['best_k']}")
        print(
            f"Validation accuracy={result['val_accuracy']:.4f}, "
            f"validation macro_f1={result['val_macro_f1']:.4f}"
        )
        print(
            f"Test accuracy={result['test_accuracy']:.4f}, "
            f"test macro_f1={result['test_macro_f1']:.4f}"
        )
        print(
            f"LaTeX row: Cosine Retrieval & {result['test_accuracy']:.2f} & "
            f"{result['test_macro_f1']:.2f} \\\\"
        )


if __name__ == "__main__":
    main()
