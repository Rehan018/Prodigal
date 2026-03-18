import pandas as pd


def stratified_sample(df: pd.DataFrame, label_col: str, n_total: int, random_state: int = 42) -> pd.DataFrame:
    """
    Create a roughly balanced stratified sample across classes.
    Assumes equal number of samples per class.
    Example:
        n_total=500 for 5 classes => 100 per class
    """
    df = df.copy()
    unique_classes = sorted(df[label_col].unique())
    n_classes = len(unique_classes)

    if n_total % n_classes != 0:
        raise ValueError(f"n_total={n_total} must be divisible by number of classes={n_classes}")

    n_per_class = n_total // n_classes
    sampled_parts = []

    for cls in unique_classes:
        class_df = df[df[label_col] == cls]
        if len(class_df) < n_per_class:
            raise ValueError(f"Not enough samples in class {cls} to draw {n_per_class} rows.")
        sampled_parts.append(class_df.sample(n=n_per_class, random_state=random_state))

    sampled_df = pd.concat(sampled_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return sampled_df