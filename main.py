from src.matrix_factorization import SVD
from src.preprocessing import (
    load_data,
    prepare_train_set,
    build_test_matrix,
    build_train_matrix,
)
from src.utils import plot_losses


if __name__ == "__main__":

    df = load_data()

    df_train, df_test = prepare_train_set(df, train_size=0.9)

    train_matrix = build_train_matrix(df_train)

    test_matrix = build_test_matrix(df_test)

    svd = SVD(epoch=10)
    svd.train(train_matrix)
    predicted = svd.predict(test_matrix[:, 0], test_matrix[:, 1])

    plot_losses(svd.losses)
