import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def find_file(filename):
    for root, dirs, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(filename)


def normalize_word(word):
    if pd.isna(word):
        return ""

    word = str(word).strip().lower()
    word = word.replace("ş", "ș").replace("ţ", "ț").replace("ã", "ă")

    word = re.sub(r"^[^\wăâîșț]+", "", word)
    word = re.sub(r"[^\wăâîșț]+$", "", word)

    return word


def z_score(values):
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / (np.nanstd(values) + 1e-9)


TRAIN_PATH = find_file("train_data.csv")
TEST_PATH = find_file("test_data.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


train["wn"] = train["word"].map(normalize_word)

test["word"] = test["token"].astype(str)
test["wn"] = test["word"].map(normalize_word)


metrics = [
    col
    for col in ["TFC", "FFD", "TRT", "TFT", "FPRT", "RRT", "skipped"]
    if col in train.columns
]


aggregation = {}

for metric in metrics:
    if metric == "skipped":
        aggregation[metric] = ["mean"]
    else:
        aggregation[metric] = ["mean", "median", "max", "std"]

group_cols = ["text_name", "wn"]

agg = train.groupby(group_cols).agg(aggregation)
agg.columns = ["_".join(col) for col in agg.columns]
agg = agg.reset_index().fillna(0)


score = 0

for col in agg.columns:
    if col == "skipped_mean":
        score -= 2.0 * z_score(agg[col])
    elif col.endswith("_mean"):
        score += 1.4 * z_score(agg[col])
    elif col.endswith("_median"):
        score += 1.0 * z_score(agg[col])
    elif col.endswith("_max"):
        score += 0.7 * z_score(agg[col])
    elif col.endswith("_std"):
        score += 1.2 * z_score(agg[col])

agg["answer_raw"] = score


test = test.merge(
    agg[group_cols + ["answer_raw"]],
    on=group_cols,
    how="left",
)


word_agg = agg.groupby("wn")["answer_raw"].mean().reset_index()
word_agg = word_agg.rename(columns={"answer_raw": "answer_word"})

test = test.merge(word_agg, on="wn", how="left")
test["answer_raw"] = test["answer_raw"].fillna(test["answer_word"])


test["word_len"] = test["wn"].map(len)
fallback = z_score(test["word_len"])

test["answer_raw"] = test["answer_raw"].fillna(pd.Series(fallback))


scaler = MinMaxScaler(feature_range=(0.01, 0.99))
test["answer"] = scaler.fit_transform(test[["answer_raw"]]).ravel()


submission = test[["datapointID", "answer"]].copy()
submission.insert(0, "subtaskID", 1)

submission.to_csv("submission_e5_eye_tracking_stats.csv", index=False)


print("Gata: submission_e5_eye_tracking_stats.csv")
print(submission.head())
print("Media:", submission["answer"].mean())
print("Std:", submission["answer"].std())