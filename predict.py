import argparse
import pandas as pd

from pathlib import Path

from utils import load_pipeline


def build_test_args():
    parser = argparse.ArgumentParser(description="Perform prediction on test data with trained pipeline")
    parser.add_argument("--pipe-path", required=True, help="Path to trained sklearn Pipeline")
    parser.add_argument("--data-path", required=True, help="Path to .csv with test data")
    parser.add_argument("--result-path", required=True, help="Path where to save predicted results")
    return parser.parse_args()


def main(args):
    pipeline_path = Path(args.pipe_path).expanduser().resolve()
    data_path = Path(args.data_path).expanduser().resolve()
    result_path = Path(args.result_path).expanduser().resolve()

    df = pd.read_csv(data_path.as_posix())

    pipeline = load_pipeline(pipeline_path.as_posix())

    predicted_target = pipeline.predict(df)

    pd.DataFrame(data={"target": predicted_target}).to_csv(result_path.as_posix())


if __name__ == "__main__":
    main(build_test_args())
