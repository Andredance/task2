import joblib

from sklearn.pipeline import Pipeline


def save_pipeline(pipeline: Pipeline, result_path: str):
    """Saves sklearn pipeline via joblib to provided result_path

    Args:
        pipeline (sklearn.pipeline.Pipeline): sklearn pipeline
        result_path (str): path where to save pipeline

    Returns:

    """
    joblib.dump(pipeline, result_path, compress=1)


def load_pipeline(pipeline_path: str) -> Pipeline:
    """Loads sklearn pipeline from pipeline_path

    Args:
        pipeline_path (str): Path to sklearn pipeline saved after training

    Returns:
        sklearn.pipeline.Pipeline: sklearn Pipeline's class object

    """
    return joblib.load(pipeline_path)
