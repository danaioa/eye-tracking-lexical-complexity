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
from scipy.stats import pearsonr


OUT = "/kaggle/working/submission_e2_lightgbm_has_eye.csv"



X_annot, X_test, y_llm, _, annot, test, feat_cols = build_features_for_test_and_annot()



has_eye_annot = annot["has_eye"].values.reshape(-1, 1).astype(float)
has_eye_test = test["has_eye"].values.reshape(-1, 1).astype(float)

X_annot = np.hstack([X_annot, has_eye_annot])
X_test = np.hstack([X_test, has_eye_test])

feat_cols = feat_cols + ["has_eye"]



kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X_annot))
test_preds = np.zeros(len(X_test))



params = dict(
    n_estimators=2500,
    learning_rate=0.015,
    num_leaves=63,
    max_depth=7,
    min_child_samples=25,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)



importances = np.zeros(len(feat_cols))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_annot), 1):
    model = LGBMRegressor(**params)

    model.fit(
        X_annot[train_idx],
        y_llm[train_idx],
        eval_set=[(X_annot[valid_idx], y_llm[valid_idx])],
        callbacks=[early_stopping(200, verbose=False)],
    )

    oof[valid_idx] = model.predict(X_annot[valid_idx])
    test_preds += model.predict(X_test) / kf.n_splits
    importances += model.feature_importances_ / kf.n_splits

    fold_score = eval_metric(y_llm[valid_idx], oof[valid_idx])
    print(f"Fold {fold}: OOF metric vs LLM = {fold_score:.2f}")



total_score = eval_metric(y_llm, oof)
print(f"\nOOF total vs LLM: {total_score:.2f}")



feature_importance = pd.DataFrame(
    {
        "feature": feat_cols,
        "importance": importances,
    }
).sort_values("importance", ascending=False)

print("\nTop 20 features:")
print(feature_importance.head(20).to_string(index=False))



print("\n=== Validare pe GOLD ===")

gold_pairs = []

for word, text_name, y_true in GOLD:
    normalized_word = normalize_word(word)

    mask = (annot["wn"] == normalized_word) & (annot["text_name"] == text_name)

    if mask.sum() == 0:
        continue

    pred = oof[mask.values].mean()
    gold_pairs.append((y_true, pred, word))

if len(gold_pairs) >= 3:
    gold_true = np.array([item[0] for item in gold_pairs])
    gold_pred = np.array([item[1] for item in gold_pairs])

    pearson_score = pearsonr(gold_true, gold_pred)[0]

    print(f"Pearson gold vs OOF-pred pe {len(gold_pairs)} cuvinte: {pearson_score:.3f}")
    print("> 0.4 = bun, ~0 = LLM zgomot, < 0 = anti-corelat")



preds = calibrate_affine(test_preds, GOLD_MEAN, GOLD_STD)

write_submission(OUT, test, preds)

print("\n[E2 - LightGBM has_eye] DONE.")