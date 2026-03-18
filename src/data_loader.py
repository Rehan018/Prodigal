from datasets import load_dataset
import pandas as pd


def load_yelp_dataset():
    """
    Load Yelp Review Full dataset from Hugging Face.
    Returns:
        train_df (pd.DataFrame), test_df (pd.DataFrame)
    """
    dataset = load_dataset("Yelp/yelp_review_full")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    return train_df, test_df