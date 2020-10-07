from pathlib import Path


class Config:
    RANDOM_SEED = 42
    ASSET_PATH = Path("./assets")
    ORIGINAL_DATASET_FILEPATH = ASSET_PATH / "original_dataset" / "udemy_courses.csv"
    DATASET_PATH = ASSET_PATH / "data"
    FEATURES_PATH = ASSET_PATH / "features"
    MODELS_PATH = ASSET_PATH / "models"
    METRICS_FILE_PATH = ASSET_PATH / "metrics.json"

