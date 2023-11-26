import argparse
import pandas as pd

from pathlib import Path
from typing import Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from utils import save_pipeline


def build_train_args():
    parser = argparse.ArgumentParser(description="Train simple linear regression")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to .csv file with train data",
        required=True
    )
    parser.add_argument(
        "--trained-pipe-path",
        type=str,
        default="trained_pipeline.pkl",
        help="Path to save pipeline after training"
    )
    return parser.parse_args()


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separates target from train data. Makes split of data to train/test.

    Args:
        df (pd.Dataframe): Source dataframe for training

    Returns:
        Tuple[pd.Dataframe, pd.Dataframe, pd.Series, pd.Series]: Train dataframe, test dataframe,
         train targets, test targets
    """
    data, targets = df.drop(["target"], axis=1), df["target"]
    data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.2,
                                                                          random_state=42)
    return data_train, data_test, targets_train, targets_test


def get_transformer_column_names(df: pd.DataFrame):
    categorical = ["8"]
    numerical = list(df.drop(categorical, axis=1).columns)
    return numerical, categorical


def train(df: pd.DataFrame) -> Pipeline:
    """Performs training of simple sklearn pipeline with StandardScaler for numerical features, generation of
    polynomial and interaction features and LinearRegression estimator.

    Args:
        df (pd.Dataframe): Source dataframe for training

    Returns:
        sklearn.pipeline.Pipeline: trained pipeline from sklearn
    """
    data_train, data_test, targets_train, targets_test = prepare_data(df)
    numerical_cols, categorical_cols = get_transformer_column_names(data_train)
    columns_transformer = ColumnTransformer(
        [
            ("numerical", StandardScaler(), numerical_cols)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("column_transformer", columns_transformer),
        ("polynomial_preprocess", PolynomialFeatures(degree=2)),
        ("estimator", LinearRegression(n_jobs=-1))
    ], verbose=True)

    pipeline.fit(data_train, targets_train)
    pred_target_train = pipeline.predict(data_train)
    pred_target_test = pipeline.predict(data_test)

    print(f"RMSE on training data: {mean_squared_error(targets_train, pred_target_train, squared=False)}")
    print(f"RMSE on test data: {mean_squared_error(targets_test, pred_target_test, squared=False)}")
    return pipeline


def main(args):
    data_path = Path(args.data_path).expanduser().resolve()
    trained_pipe_path = Path(args.trained_pipe_path).expanduser().resolve()

    if not data_path.exists():
        raise ValueError("Please, provide valid path to .csv file that exists!")

    df = pd.read_csv(data_path.as_posix())
    trained_pipeline = train(df)
    save_pipeline(trained_pipeline, trained_pipe_path.as_posix())


if __name__ == "__main__":
    main(build_train_args())
