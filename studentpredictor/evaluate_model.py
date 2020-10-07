import json
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

from config import Config

# Read in X and y test data
X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

# Read in model specs
model = pickle.load(open(str(Config.MODELS_PATH / "model.pickle"), "rb"))

# R2
r_squared = model.score(X_test, y_test)

# Predict with train model specification
y_pred = model.predict(X_test)

# Print RMSE
rmse = mean_squared_error(y_test, y_pred)

# Dump as json (R2 and RMSE)
with open(str(Config.METRICS_FILE_PATH), "w") as outfile:
    json.dump(dict(r_squared=r_squared, rmse=rmse), outfile)

