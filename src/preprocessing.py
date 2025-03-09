import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data/ml-100k.data")
)


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    columns_names = ["user_id", "item_id", "rating", "timestamp"]
    return pd.read_csv(filepath, sep="\t", names=columns_names).drop(
        columns="timestamp"
    )


def prepare_train_set(
    df: pd.DataFrame, train_size: float = 0.8, random_state: int = 42
) -> {pd.DataFrame, pd.DataFrame}:
    df_train, df_test = train_test_split(
        df, train_size=train_size, random_state=random_state
    )
    return df_train, df_test


def build_train_matrix(df_train: pd.DataFrame) -> np.ndarray:
    return df_train.pivot(
        index="user_id", columns="item_id", values="rating"
    ).to_numpy()


def build_test_matrix(df_test: pd.DataFrame) -> np.ndarray:
    return df_test.to_numpy()