# Importing all the libraries needed
import pandas as pd
import os
from sklearn.compose import (
    ColumnTransformer,
    make_column_transformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import FeatureUnion, make_union
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

# translating jupyter notebook in the terminal
# copying the model from the translated py script

# Defining needed functions
def ffill_missing(ser):
    return ser.fillna(method="ffill")


def is_weekend(data):
    return data["dteday"].dt.day_name().isin(["Saturday", "Sunday"]).to_frame()


def year(data):
    # Our reference year is 2011, the beginning of the training dataset
    return (data["dteday"].dt.year - 2011).to_frame()


# data reading and assessment

DIRECTORY_WHERE_THIS_FILE_IS = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "hour.csv")


def train_and_persist():
    # Data preparation
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    X = df.drop(columns=["instant", "cnt", "casual", "registered"])
    y = df["cnt"]

    import numpy as np

    # filler
    ffiller = FunctionTransformer(ffill_missing)

    # Weather encoder
    weather_enc = make_pipeline(
        ffiller,
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
        ),
    )
    # column transformer
    ct = make_column_transformer(
        (ffiller, make_column_selector(dtype_include=np.number)),
        (weather_enc, ["weathersit"]),
    )

    # Pre processing
    preprocessing = FeatureUnion(
        [
            ("is_weekend", FunctionTransformer(is_weekend)),
            ("year", FunctionTransformer(year)),
            ("column_transform", ct),
        ]
    )
    # model
    reg = Pipeline(
        [("preprocessing", preprocessing), ("model", RandomForestRegressor())]
    )
    reg
    X_train, y_train = X.loc[X["dteday"] < "2012-10"], y.loc[X["dteday"] < "2012-10"]
    X_test, y_test = X.loc["2012-10" <= X["dteday"]], y.loc["2012-10" <= X["dteday"]]
    reg.fit(X_train, y_train)

    # Scoring:
    reg.score(X_test, y_test)

    # Making predictions:
    y_pred = reg.predict(X_test)

    # saving the model using joblib
    import joblib as joblib
    from joblib import dump, load

    model_name = "train_persist_predict.joblib"
    joblib.dump(reg, model_name)


# Trains the model and saves it to `model.joblib`
def predict(dteday, hr, weathersit, temp, atemp, hum, windspeed):
    import joblib as joblib
    from joblib import dump, load

    reg = joblib.load("train_persist_predict.joblib")
    preds = reg.predict(
        pd.DataFrame(
            [[pd.to_datetime(dteday), hr, weathersit, temp, atemp, hum, windspeed]],
            columns=["dteday", "hr", "weathersit", "temp", "atemp", "hum", "windspeed"],
        )
    )
    return preds[0]


# file reformatted with "black ." using the terminal
