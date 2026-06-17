import sys
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, "/kaggle/working")

try:
    from lightgbm import LGBMRegressor, early_stopping
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])
    from lightgbm import LGBMRegressor, early_stopping

from sklearn.model_selection import KFold


OUT = "/kaggle/working/submission_e3_lightgbm_quantile_shrink.csv"


X_annot, X_test, y_llm, _, annot, test, feat_cols = build_features_for_test_and_annot()


kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X_annot))
test_preds = np.zeros(len(X_test))


params = dict(
    n_estimators=1500,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


for fold, (train_idx, valid_idx) in enumerate(kf.split(X_annot), 1):
    model = LGBMRegressor(**params)

    model.fit(
        X_annot[train_idx],
        y_llm[train_idx],
        eval_set=[(X_annot[valid_idx], y_llm[valid_idx])],
        callbacks=[early_stopping(150, verbose=False)],
    )

    oof[valid_idx] = model.predict(X_annot[valid_idx])
    test_preds += model.predict(X_test) / kf.n_splits

    fold_score = eval_metric(y_llm[valid_idx], oof[valid_idx])
    print(f"Fold {fold}: OOF metric vs LLM = {fold_score:.2f}")


total_score = eval_metric(y_llm, oof)

print(
    f"\nOOF total vs LLM target: {total_score:.2f} "
    "(atentie: evaluarea este fata de targetul LLM, nu fata de gold real)"
)


preds = calibrate_quantile(test_preds)


SHRINK = 0.7

preds = preds.mean() + SHRINK * (preds - preds.mean())
preds = np.clip(preds, 0, 1)


write_submission(OUT, test, preds)

print("\n[E3 - LightGBM quantile shrink] DONE.")
print(f"SHRINK = {SHRINK}")