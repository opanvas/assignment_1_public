# Importing all the libraries needed
import pandas as pd
from ie_bike_model.model import train_and_persist, predict
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
from joblib import dump, load

# should this be defined or imported lol
from ie_bike_model.model import train_and_persist

# translating jupyter notebook in the terminal
# copying the model from the translated py script

# data reading and assessment

df = pd.read_csv("hour.csv", parse_dates=["dteday"])
df.head()

df.info()

# Data preparation

X = df.drop(columns=["instant", "cnt", "casual", "registered"])
y = df["cnt"]

# Defining and applying the needed functions

def ffill_missing(ser):
    return ser.fillna(method="ffill")

ffiller = FunctionTransformer(ffill_missing)

# Weather encoder

weather_enc = make_pipeline(
    ffiller,
    OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
    ),
)
weather_enc

# Date transformations

def is_weekend(data):
    return (
        data["dteday"]
        .dt.day_name()
        .isin(["Saturday", "Sunday"])
        .to_frame()
    )

def year(data):
    # Our reference year is 2011, the beginning of the training dataset
    return (data["dteday"].dt.year - 2011).to_frame()

# Pre processing

preprocessing = FeatureUnion([
    ("is_weekend", FunctionTransformer(is_weekend)),
    ("year", FunctionTransformer(year)),
    ("column_transform", ct)
])
preprocessing


# Defining the model

reg = Pipeline([("preprocessing", preprocessing), ("model", RandomForestRegressor())])
reg


X_train, y_train = X.loc[X["dteday"] < "2012-10"], y.loc[X["dteday"] < "2012-10"]
X_test, y_test = X.loc["2012-10" <= X["dteday"]], y.loc["2012-10" <= X["dteday"]]


reg.fit(X_train, y_train)


# In[19]:


reg.score(X_test, y_test)


# Making predictions:


y_pred = reg.predict(X_test)

print(y_pred)

#

X_train.head()


train_and_persist()  # Trains the model and saves it to `model.joblib`
predict( 
    dteday="2012-11-01",
    hr=10,
    weathersit="Clear, Few clouds, Partly cloudy, Partly cloudy"
    temp=0.3,
    atemp=0.31,
    hum=0.8,
    windspeed=0.0)



# In[27]:


reg.predict(pd.DataFrame([[
    pd.to_datetime("2012-11-01"),
    10,
    "Clear, Few clouds, Partly cloudy, Partly cloudy",
    0.3,
    0.31,
    0.8,
    0.0,
]], columns=[
    'dteday',
    'hr',
    'weathersit',
    'temp',
    'atemp',
    'hum',
    'windspeed'
]))



# file reformatted with "black ." using the terminal
