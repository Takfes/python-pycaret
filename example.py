import numpy as np
import pandas as pd
from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce

# def glimpse(df):
#     return pd.DataFrame(
#         {"dtypes": df.dtypes, "missing": df.isnull().sum(), "nunique": df.nunique()}
#     )
# glimpse(df)

df = get_data("automobile")

# df.loc[[0, 5, 131, 141], "make"] = np.nan
# df.loc[[1, 6, 132, 142], "body-style"] = np.nan
# df.loc[[2, 7, 133, 143], "normalized-losses"] = np.nan

df_train = df.head(130)
df_valid = df.loc[130:]

tscv = TimeSeriesSplit(test_size=int(df_train.shape[0] / 10), n_splits=5)

TARGET = "price"

columns_numeric = ["normalized-losses"]
columns_cardinal = ["make"]
columns_onehot = ["body-style"]

pipe_numeric = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("transformer", PowerTransformer(method="yeo-johnson")),
    ]
)

pipe_cardinal = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cardinal", ce.CatBoostEncoder()),
    ]
)

pipe_categorical = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", pipe_numeric, columns_numeric),
        ("categorical", pipe_categorical, columns_onehot),
        ("cardinal", pipe_cardinal, columns_cardinal),
    ],
    remainder="drop",
)

# xx = preprocessor.fit_transform(df_train.drop('price',axis=1),df_train['price'])
# pd.DataFrame(xx).isnull().sum()

rgr = RegressionExperiment()

rgr.setup(
    data=df_train,
    target="price",
    test_data=df_valid,
    preprocess=False,
    custom_pipeline=preprocessor,
    data_split_shuffle=False,
    fold_strategy=tscv,
)
