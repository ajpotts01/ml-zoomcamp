# Standard lib
import pickle
from typing import Union, Tuple

# Non-standard lib
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# region Reference variables/constants
DATA_PATH = "../data/telco.csv"
OUTPUT_PATH = "../models/model_reg=%s.bin"

numeric_variables = ["tenure", "monthlycharges", "totalcharges"]
categorical_variables = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]

# Parameters decided from the followalong video - oddly, the follow-along differed to Alexey
# The winning reg factor was 0.1. But then porting the code to this notebook and following along again, 1 is the winner.
# Must have mixed up something in the variables in the (very very long) previous notebook.
regularization_factor = 1
n_splits = 5
final_reg_factor = 0.1

# endregion Reference variables/constants


# region Column standardisation methods
def string_transformations(
    target: Union[pd.core.strings.accessor.StringMethods, pd.core.indexes.base.Index]
) -> Union[pd.core.series.Series, pd.core.indexes.base.Index]:
    """
    Stage 1 cleaning for this churn prediction:
    - Lower case for everything
    - Spaces replaced by underscores

    Can work on either Pandas indices (e.g. column headers) or Pandas series (e.g. row data)

    :param StringMethods | Index target: the target row or column to standardise
    :return StringMethods | Index result: the standardised row or column

    Note the return types are using typing since this was written pre-3.10.
    """
    result = target.str.lower().str.replace(" ", "_")

    return result


def standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new.columns = string_transformations(target=df_new.columns)

    object_filter = df_new.dtypes == type(object)

    categorical_columns = list(df_new.dtypes[object_filter].index)
    df_new[categorical_columns] = df_new[categorical_columns].apply(
        func=string_transformations, axis=1
    )
    return df_new


def standardise_float(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    new_col = pd.to_numeric(df[col_name], errors="coerce")

    df_new = df.copy()
    df_new[col_name] = new_col.fillna(0)

    return df_new


def encode_labels(df: pd.DataFrame, col_name: str, encode_value: str) -> pd.DataFrame:
    new_col = (df[col_name] == encode_value).astype(int)

    df_new = df.copy()
    df_new[col_name] = new_col

    return df_new


# endregion Column standardisation


# region Training and prediction methods
def train(
    df: pd.DataFrame,
    y_train: np.ndarray,
    categorical: list[str],
    numerical: list[str],
    regularization_factor: float = 1.0,
) -> Tuple[DictVectorizer, LogisticRegression]:
    dict_train: dict = df[categorical + numerical].to_dict(orient="records")

    vect = DictVectorizer(sparse=False)
    X_train = vect.fit_transform(dict_train)

    model = LogisticRegression(C=regularization_factor, max_iter=1000)
    model.fit(X_train, y_train)

    return vect, model


def predict(
    df: pd.DataFrame,
    vect: DictVectorizer,
    model: LogisticRegression,
    categorical: list[str],
    numerical: list[str],
) -> np.ndarray:
    dict_predict: dict = df[categorical + numerical].to_dict(orient="records")

    X = vect.transform(dict_predict)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


def training_loop(
    df: pd.DataFrame,
    categorical: list[str],
    numerical: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    regularization_factor: float,
) -> np.ndarray[float]:
    df_train_next = df.iloc[train_idx]
    df_val_next = df.iloc[val_idx]

    y_train_next = df_train_next["churn"].values
    y_val_next = df_val_next["churn"].values

    vect_next, mdl_next = train(
        df=df_train_next,
        y_train=y_train_next,
        categorical=categorical,
        numerical=numerical,
        regularization_factor=regularization_factor,
    )
    y_pred_next = predict(
        df=df_val_next,
        vect=vect_next,
        model=mdl_next,
        categorical=categorical,
        numerical=numerical,
    )

    auc = roc_auc_score(y_val_next, y_pred_next)

    return auc


# endregion Training and Predictions

# region Dataset prep
print("Loading and preparing dataset...")
df_churn = pd.read_csv(DATA_PATH)
df_churn_standardised = standardise_strings(df_churn)
df_fixed_charges = standardise_float(df=df_churn_standardised, col_name="totalcharges")
df_encoded_labels = encode_labels(
    df=df_fixed_charges, col_name="churn", encode_value="yes"
)

df_full_train, df_test = train_test_split(
    df_encoded_labels, test_size=0.2, random_state=1
)
y_test = df_test["churn"].values
# endregion Dataset prep

# region Some scoring tests before training and saving final model
print("Doing k-fold validation")
k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []

scores = [
    training_loop(
        df=df_full_train,
        categorical=categorical_variables,
        numerical=numeric_variables,
        train_idx=train_idx,
        val_idx=val_idx,
        regularization_factor=regularization_factor,
    )
    for train_idx, val_idx in k_fold.split(df_full_train)
]

print("Reg=%s %.3f +- %.3f" % (regularization_factor, np.mean(scores), np.std(scores)))
# endregion Some scoring tests before training and saving final model

# region Final training
# Ah... when evaluating model earlier, must have been using wrong dataframes.
# The AUC is clearly higher when using regularization factor of 0.1, so sticking with that.
print("Final training...")
vect_final, mdl_final = train(
    df=df_full_train,
    y_train=df_full_train["churn"].values,
    categorical=categorical_variables,
    numerical=numeric_variables,
    regularization_factor=final_reg_factor,
)
y_pred_final = predict(
    df=df_test,
    vect=vect_final,
    model=mdl_final,
    categorical=categorical_variables,
    numerical=numeric_variables,
)

auc = roc_auc_score(y_test, y_pred_final)
print(f"{auc=}")
# endregion Final training

# region Saving model
# Time to save!
# Use pickle from std lib
file_path = OUTPUT_PATH % final_reg_factor
print(f"Saving to {file_path}")
with open(file_path, "wb") as file_model_output:  # wb is write, binary
    pickle.dump((vect_final, mdl_final), file_model_output)
# endregion Saving model
