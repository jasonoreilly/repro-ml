from datetime import date
import pandas as pd
from config import Config


def extract_features_on_time(df):
    """
    Extract features based on timestamp and days since published
    :param df: dataframe on which to
    :return: dataframe with pre-specified
    """
    df["published_timestamp"] = pd.to_datetime(df.published_timestamp).dt.date
    df["days_since_published"] = (date.today() - df.published_timestamp).dt.days

    return df[["num_lectures", "price", "days_since_published", "content_duration"]]


# Create directories according to config
Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

# Read in data
train_df = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
test_df = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))

# Extract additional features
train_features = extract_features_on_time(train_df)
test_features = extract_features_on_time(test_df)

# Save data with new features
train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

# Create file with only target variable (number subscribers) and save to directory according to config file
train_df.num_subscribers.to_csv(
    str(Config.FEATURES_PATH / "train_labels.csv"), index=None
)
test_df.num_subscribers.to_csv(
    str(Config.FEATURES_PATH / "test_labels.csv"), index=None
)

