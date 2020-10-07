import gdown
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import Config

# Set random seed according to config file
np.random.seed(Config.RANDOM_SEED)

# Create directories according to config
Config.ORIGINAL_DATASET_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Download data from Google Drive
gdown.download(
    "https://drive.google.com/uc?id=1gkYBOIMm8pAGunRoI3OzQHQrgOdaRjfC",         # specify download link
    str(Config.ORIGINAL_DATASET_FILEPATH),                                      # specify where to download to
)

# Read in downloaded file as pandas df
df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILEPATH))

# Split data into train and test
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=Config.RANDOM_SEED,
)

# Save files as train and test to path as specified in config file
df_train.to_csv(str(Config.DATASET_PATH / "train.csv"), index=None)
df_test.to_csv(str(Config.DATASET_PATH / "test.csv"), index=None)


