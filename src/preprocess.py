import pandas as pd


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw label column and create stars column (1 to 5).
    Keeps only required columns: text, label_raw, stars
    Safe to re-run on already standardized data.
    """
    df = df.copy()

    # Accept both raw ('label') and already-standardized ('label_raw')
    if "label" in df.columns:
        df = df.rename(columns={"label": "label_raw"})
    elif "label_raw" not in df.columns:
        raise ValueError(
            f"Expected column 'label' or 'label_raw' not found. "
            f"Columns present: {list(df.columns)}"
        )

    if "text" not in df.columns:
        raise ValueError(
            f"Expected column 'text' not found. "
            f"Columns present: {list(df.columns)}"
        )

    df["stars"] = df["label_raw"] + 1
    df = df[["text", "label_raw", "stars"]]

    return df


def add_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add text-based statistics columns.
    """
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["char_length"] = df["text"].apply(len)
    df["word_length"] = df["text"].apply(lambda x: len(x.split()))
    return df


def get_quality_checks(df: pd.DataFrame) -> dict:
    """
    Return dataset quality checks.
    """
    checks = {
        "rows": len(df),
        "null_text": int(df["text"].isnull().sum()),
        "null_label_raw": int(df["label_raw"].isnull().sum()),
        "null_stars": int(df["stars"].isnull().sum()),
        "empty_reviews": int((df["text"].astype(str).str.strip() == "").sum()),
        "duplicate_reviews": int(df.duplicated(subset=["text"]).sum()),
    }
    return checks


def get_class_distribution(df: pd.DataFrame) -> dict:
    """
    Return class distribution for stars 1-5.
    """
    dist = df["stars"].value_counts().sort_index().to_dict()
    return dist


def get_length_summary(df: pd.DataFrame) -> dict:
    """
    Return summary stats for character and word lengths.
    """
    summary = {
        "avg_char_length": round(df["char_length"].mean(), 2),
        "avg_word_length": round(df["word_length"].mean(), 2),
        "min_word_length": int(df["word_length"].min()),
        "max_word_length": int(df["word_length"].max()),
    }
    return summary


def build_dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """
    Create a readable text summary of dataset checks and stats.
    """
    train_checks = get_quality_checks(train_df)
    test_checks = get_quality_checks(test_df)

    train_dist = get_class_distribution(train_df)
    test_dist = get_class_distribution(test_df)

    train_len = get_length_summary(train_df)
    test_len = get_length_summary(test_df)

    summary = f"""
YELP REVIEW FULL - DATASET SUMMARY

TRAIN DATASET
-------------
Rows: {train_checks['rows']}
Null text: {train_checks['null_text']}
Null label_raw: {train_checks['null_label_raw']}
Null stars: {train_checks['null_stars']}
Empty reviews: {train_checks['empty_reviews']}
Duplicate reviews: {train_checks['duplicate_reviews']}
Class distribution: {train_dist}
Length summary: {train_len}

TEST DATASET
------------
Rows: {test_checks['rows']}
Null text: {test_checks['null_text']}
Null label_raw: {test_checks['null_label_raw']}
Null stars: {test_checks['null_stars']}
Empty reviews: {test_checks['empty_reviews']}
Duplicate reviews: {test_checks['duplicate_reviews']}
Class distribution: {test_dist}
Length summary: {test_len}
""".strip()

    return summary