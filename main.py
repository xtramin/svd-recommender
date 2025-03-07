import os
import pandas as pd
from src.matrix_factorization import SVD

if __name__ == "__main__":
    column_names = ["user_id", "item_id", "rating", "timestamp"]

    print(os.path.dirname(__file__))
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data/ml-100k.data"),
        sep="\t",
        names=column_names,
    ).drop(columns="timestamp")

    matrix = df.pivot(index="user_id", columns="item_id", values="rating").to_numpy()

    u, v = SVD().train(X=matrix)

    print(u.shape, v.shape)
