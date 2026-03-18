import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(df: pd.DataFrame, true_col: str = "stars", pred_col: str = "stars_pred") -> dict:
    result = {
        "total_rows": len(df),
        "json_compliance_rate": 0.0,
        "valid_prediction_rows": 0,
        "accuracy": None,
        "macro_f1": None,
    }

    if "json_valid" not in df.columns:
        raise ValueError("Expected column 'json_valid' not found.")

    result["json_compliance_rate"] = round(df["json_valid"].mean(), 4)

    valid_df = df[df["json_valid"] == True].copy()
    result["valid_prediction_rows"] = len(valid_df)

    if len(valid_df) == 0:
        return result

    y_true = valid_df[true_col]
    y_pred = valid_df[pred_col]

    result["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    result["macro_f1"] = round(f1_score(y_true, y_pred, average="macro"), 4)

    return result


def print_metrics(title: str, metrics: dict):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")