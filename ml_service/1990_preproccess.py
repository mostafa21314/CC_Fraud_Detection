from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
CSV_PATH   = "data/final_dataset.csv"
TARGET_COL = "Is Fraud?"

df = pd.read_csv(CSV_PATH)
TEST_SIZE    = 0.20 #20% of untouched training data from testing set

# Change target column to binary
if df[TARGET_COL].dtype == object:
    df[TARGET_COL] = df[TARGET_COL].map({'Yes': 1, 'No': 0}).fillna(df[TARGET_COL]).astype(int)

df["Is Fraud?"] = df["Is Fraud?"].astype("int8")

#  Normalize seconds_1990
scaler = MinMaxScaler()
df['seconds_1990'] = scaler.fit_transform(df[['seconds_1990']])

# Convert all numeric columns to float32 to save RAM
for col in df.select_dtypes(include=["float64", "int64"]).columns:
    if col != "Is Fraud?":
        df[col] = df[col].astype("float32")

df.drop("seconds", axis=1, inplace=True)
df.drop("Amount_float", axis=1, inplace=True)


df.to_csv("final_dataset_processed.csv", index=False)


