import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import Config

# Create directory
Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Read in X and y data
X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

# Specify and fit model
model = LinearRegression()
model = model.fit(X_train, y_train.to_numpy().ravel())

pickle.dump(model, open(str(Config.MODELS_PATH / "model.pickle"), "wb"))

