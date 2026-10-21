# =============================================================================
# 02_MODELLING.PY
#
# Purpose: Evaluate four supervised ML models for classifying prediabetes risk
#          (low-risk cluster C2 vs high-risk clusters C3/C5/C6) from blood DNA
#          methylation beta values.
#
# Reads:  Data/interim/methylation_filtered.parquet   (built by 01_prepare_data.py)
# Writes: results/  (figures + tables)
#
# This is the iterative modelling script. It never touches the 1.7 GB source
# CSV; that was a one-time operation handled by the preparation script.
#
# THE CENTRAL DESIGN CONSTRAINT:
# Feature selection and scaling are wrapped INSIDE scikit-learn Pipelines, so
# they are re-fitted from scratch on each cross-validation fold. If they were
# applied to the whole dataset up front, information from the held-out data
# would leak into the choice of features and inflate every result. With ~194k
# features and only 187 samples, that leakage would be severe.
# =============================================================================
 

# %%
# =============================================================================
# STEP 3: DISCOVERY-ONLY SUBSET + TRAIN/TEST SPLIT
# =============================================================================
import numpy as np
np.random.seed(42)

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

# The repository root (the folder above code/). When the "# %%" cells are run
# interactively, __file__ is not defined, so the working directory is used
# instead: open the repository root in VS Code before running the cells.
BASE_DIR    = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
INTERIM_DIR = BASE_DIR / "Data" / "interim"

# Every random process below (the split, CV fold assignment, model internals)
# is pinned to this seed. This makes the entire pipeline deterministic:
# re-running produces identical results rather than a slightly different set
# each time.
RANDOM_STATE = 42

# --- 1. Load the cached dataset ---------------------------------------------
# Parquet loads in seconds. Contains all 333 samples, ~289k variance-filtered
# CpG columns, plus four metadata columns (risk, cluster, cohort, platform).
data = pd.read_parquet(INTERIM_DIR / "methylation_filtered.parquet")
print("Loaded cache:", data.shape)

# --- 2. RESTRICT TO THE DISCOVERY COHORT ------------------------------------
# This is the single most important analytical decision in the project.
#
# The dataset contains a perfect confound: all 187 Discovery samples were run
# on EPIC v1 and include all 50 low-risk individuals, while all 146 Replication
# samples were run on EPIC v2 and are 100% high-risk. Consequently, "measured
# on v2" predicts "high-risk" with perfect accuracy.
#
# Array versions differ in their technical signature, and ML models are
# effective at detecting exactly that kind of batch effect. Trained on all 333
# samples, a model could achieve high accuracy by recognising the chip rather
# than the biology — learning nothing about methylation.
#
# Restricting to the Discovery cohort eliminates the confound (a single
# platform remains, so no shortcut exists) and improves class balance from
# 15%/85% to 27%/73%. The cost is 146 samples, but those samples were
# contributing confound, not information.
data = data[data["cohort"] == "Discovery"].copy()
print("Discovery only:", data.shape)
print("Class balance:\n", data["risk"].value_counts(), "\n")

# --- 3. Separate features from metadata -------------------------------------
# X = the CpG beta values the models learn from.
# y = the binary target (0 = low-risk C2, 1 = high-risk C3/C5/C6).
# The metadata columns are dropped from X — 'cluster' would leak the answer
# outright, and 'platform'/'cohort' are now constant within Discovery.
meta_cols = ["risk", "cluster", "cohort", "platform"]
X = data.drop(columns=meta_cols)
y = data["risk"]

# --- 4. Re-apply the variance filter WITHIN Discovery ------------------------
# The original filter (in 01_prepare_data.py) measured variance across all 333
# samples spanning both array versions. Some CpGs survived that filter not
# because they differ between individuals, but because they differ between
# platforms.
#
# Recomputing variance within a single platform removes those probes. The
# reduction of a further ~95k sites (289k → 194k) is itself evidence of how
# much apparent variation was technical rather than biological.
vt = VarianceThreshold(threshold=0.001)
X_arr = vt.fit_transform(X)
X = pd.DataFrame(X_arr, index=X.index, columns=X.columns[vt.get_support()])
print("After Discovery-only variance filter:", X.shape)

# --- 5. Hold out 20% as a genuinely unseen test set --------------------------
# stratify=y ensures both partitions preserve the ~27% low-risk proportion.
# Without stratification, a random draw could place an unrepresentative number
# of low-risk samples in the test set — a tangible risk when only 50 exist.
#
# This 20% is now sealed. It plays no part in feature selection, model fitting,
# or tuning; it is used exactly once, in Step 6.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print(f"Features (CpGs):  {X_train.shape[1]}\n")
print("Training class balance:\n", y_train.value_counts(), "\n")
print("Test class balance:\n", y_test.value_counts())


# %%
# =============================================================================
# STEP 4a: LOGISTIC REGRESSION BASELINE  [optional — superseded by Step 4b]
# =============================================================================
#
# This block established the leakage-safe pipeline pattern that every
# subsequent model follows. Step 4b now runs Logistic Regression as one of its
# four models, making this block redundant for producing results. It is
# retained as a record of how the pipeline structure was developed and
# verified incrementally.

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# --- The pipeline ---
# Bundling scaler, selector, and classifier into one object is what makes
# cross-validation honest: when cross_validate holds out a fold, that fold is
# excluded from fitting ALL THREE stages, not just the final classifier.
lr_pipeline = Pipeline([
      # 1. Standardise each CpG to mean 0, SD 1. Logistic Regression penalises
    #      coefficients, so features on different scales would be penalised
    #      unequally. (Tree models are invariant to scaling, but it does them
    #      no harm.)
    ("scaler", StandardScaler()),

    # 2. LASSO feature selection. An L1 penalty shrinks uninformative
    #    coefficients to exactly zero; SelectFromModel then retains only the
    #    survivors. This is what makes ~194k features computationally tractable.
    ("select", SelectFromModel(
        LogisticRegression(
            penalty="l1", solver="liblinear",   # liblinear handles L1 properly
            C=0.1, class_weight="balanced", max_iter=1000,
            random_state=42, 
        )
    )),

    # 3. The classifier itself.
    ("clf", LogisticRegression(
        class_weight="balanced",   # up-weights the minority class (27% low-risk)
        max_iter=1000,             # high-dimensional data needs room to converge
        random_state=42,
    )),
])

# 5-fold stratified CV: each fold preserves the class balance, so no fold ends
# up with too few low-risk samples to score meaningfully.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "f1", "roc_auc"]

# Note: cross-validation runs on the TRAINING set only. The test set is untouched.
print("Running 5-fold cross-validation for Logistic Regression...")
cv_results = cross_validate(
    lr_pipeline, X_train, y_train,
    cv=cv, scoring=scoring,
    return_train_score=False,
)

# Mean and SD across folds are both reported. The SD is as informative as the
# mean here — with ~40 low-risk training samples split five ways, each fold's
# estimate rests on a handful of cases, so fold-to-fold variability is
# substantial and diagnostically meaningful.
print("\nLogistic Regression — cross-validated performance (training set):")
for metric in scoring:
    scores = cv_results[f"test_{metric}"]
    print(f"  {metric:10s}: {scores.mean():.3f}  (+/- {scores.std():.3f})")


# %%
# =============================================================================
# STEP 4b: ALL FOUR MODELS — UNTUNED BASELINE COMPARISON
# =============================================================================
#
# Runs LR, Random Forest, SVM, and XGBoost through an IDENTICAL pipeline
# structure with default hyperparameters. Identical structure is the point:
# same scaler, same selector, same folds, same seed — so any difference in
# performance is attributable to the algorithm, not the experimental setup.
#
# This serves as the "before tuning" reference point against which Step 5 is
# compared.

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

RANDOM_STATE = 42

def make_selector():
    """
     The shared LASSO selection stage.
 
    Defined as a function rather than a single shared object because each
    model requires its own fresh instance — reusing one fitted selector
    across pipelines would carry state between them. Each call returns an
    unfitted copy.
    """
    return SelectFromModel(
        LogisticRegression(
            penalty="l1", solver="liblinear",
            C=0.1, class_weight="balanced", max_iter=1000,
            random_state=42, 
        )
    )

# --- Handling class imbalance -----------------------------------------------
# Three of the four models accept class_weight="balanced", which automatically
# up-weights the minority class in inverse proportion to its frequency.
# XGBoost uses a separate parameter, scale_pos_weight, which expects the ratio
# directly; it is therefore computed here to ensure equivalent treatment.
n_high = (y_train == 1).sum()
n_low  = (y_train == 0).sum()
pos_weight = n_high / n_low     # ~2.7 after the Discovery restriction

models = {
    # Interpretable linear baseline. Its coefficients map directly onto CpGs,
    # which is why the proposal specified it despite likely lower accuracy.
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
    ),
    # Ensemble of decision trees. Handles non-linear relationships and copes
    # well with noisy predictors — both common characteristics of methylation data.
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE,
    ),
    # Often strong when features far outnumber samples (p >> n), which is
    # precisely this dataset's structure. probability=True is required for
    # ROC-AUC computation but makes SVM noticeably slower (it runs internal
    # CV to calibrate probabilities).
    "SVM": SVC(
        class_weight="balanced", probability=True, random_state=RANDOM_STATE,
    ),
    # Gradient boosting; typically strong on structured data. eval_metric is set
    # explicitly to suppress a deprecation warning, not for any modelling reason.
    "XGBoost": XGBClassifier(
        scale_pos_weight=pos_weight, eval_metric="logloss", random_state=RANDOM_STATE,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "f1", "roc_auc"]

# --- Run every model through the same pipeline shape ------------------------
results = []
for name, clf in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("select", make_selector()),
        ("clf", clf),
    ])

    print(f"Running 5-fold CV: {name} ...")
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv, scoring=scoring,
    )

    results.append({
        "Model": name,
        "Accuracy":     cv_results["test_accuracy"].mean(),
        "Accuracy_std": cv_results["test_accuracy"].std(),
        "F1":           cv_results["test_f1"].mean(),
        "F1_std":       cv_results["test_f1"].std(),
        "ROC_AUC":      cv_results["test_roc_auc"].mean(),
        "ROC_AUC_std":  cv_results["test_roc_auc"].std(),
    })

results_df = pd.DataFrame(results).set_index("Model")
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

# INTERPRETING THIS TABLE: ROC-AUC is the primary metric. Accuracy and F1 are
# both inflated by the majority class; a model predicting "high-risk" for
# every sample achieves 73% accuracy while being clinically useless. AUC
# measures actual discrimination between the two groups; 0.5 denotes chance.
print("\n=== Cross-validated comparison (training set) ===\n")
print(results_df[["Accuracy", "F1", "ROC_AUC"]])
print("\n(Standard deviations across folds:)\n")
print(results_df[["Accuracy_std", "F1_std", "ROC_AUC_std"]])


# %%
# =============================================================================
# STEP 5: TWO-STAGE FEATURE SELECTION (LASSO -> RF) + HYPERPARAMETER TUNING
# =============================================================================
#
# Two changes from Step 4b:
#   1. A second selection stage (Random Forest importance) after LASSO,
#      completing the multi-step strategy specified in the methodology.
#   2. GridSearchCV to tune each model's hyperparameters.
#
# Both tuning and selection occur entirely within the training set via
# cross-validation. The test set remains untouched.

from sklearn.model_selection import GridSearchCV

def make_lasso_selector():
    """Stage 1: L1 regularisation drops CpGs with zero coefficients."""
    return SelectFromModel(
        LogisticRegression(
            penalty="l1", solver="liblinear",
            C=0.1, class_weight="balanced", max_iter=1000,
            random_state=42, 
        )
    )

def make_rf_selector():
    """
    Stage 2: Rank the LASSO survivors by Random Forest importance and retain
    those exceeding the mean importance threshold.
 
    The two stages are complementary rather than redundant: LASSO is linear and
    penalises correlated features (methylation at neighbouring CpGs is often
    highly correlated), whereas RF importance captures non-linear contributions.
    This constitutes the multi-step approach the literature favours over
    single-step filtering.
    """
    return SelectFromModel(
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        threshold="mean",     # keep features scoring above mean importance
    )

pos_weight = (y_train == 1).sum() / (y_train == 0).sum()

# --- The search grids -------------------------------------------------------
# Deliberately small. Every combination is evaluated across 5 folds, each fold
# refitting both selection stages on ~194k features — so grid size drives
# runtime directly. These ranges cover the parameters that matter most per
# model.
#
# The "clf__" prefix tells GridSearchCV which pipeline STEP a parameter belongs
# to: "clf__C" targets the C parameter of the step named 'clf'. Without it,
# GridSearchCV could not disambiguate between the classifier's and the
# selector's parameters.
tuning_configs = {
    "Logistic Regression": {
        "estimator": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        # C = inverse regularisation strength. Smaller values produce simpler,
        # more constrained models. With p >> n, heavy regularisation often
        # performs best.
        "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
         # More trees yield more stable estimates; max_depth caps how far each
         # tree can split, thereby limiting overfitting on a small sample.
        "param_grid": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 10]},
    },
    "SVM": {
        "estimator": SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        # 'linear' vs 'rbf' is the key comparison: high-dimensional genomic
        # data is often linearly separable, so the linear kernel can outperform
        # the more flexible rbf kernel despite being simpler.
        "param_grid": {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
    },
    "XGBoost": {
        "estimator": XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", random_state=RANDOM_STATE),
        # learning_rate controls how much each tree corrects the last;
        # max_depth controls per-tree complexity. Lower values of both produce
        # more conservative models.
        "param_grid": {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 6], "clf__learning_rate": [0.05, 0.1]},
    },
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

best_models = {}
for name, cfg in tuning_configs.items():
    # Note the pipeline now has FOUR stages. All four are refitted on each fold.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso_select", make_lasso_selector()),   # Stage 1: linear, L1
        ("rf_select",    make_rf_selector()),      # Stage 2: non-linear, importance
        ("clf", cfg["estimator"]),
    ])

    print(f"Tuning {name} (two-stage selection) ...")
    search = GridSearchCV(
        pipeline, param_grid=cfg["param_grid"],
        cv=cv,
        scoring="roc_auc",   # Optimise for discrimination, NOT accuracy; optimising
                             # accuracy here would reward majority-class guessing
        n_jobs=1,            # Single-threaded to ensure deterministic results
        refit=True,          # refit the winner on the full training set
    )
    search.fit(X_train, y_train)

    # best_estimator_ is the winning pipeline, already refitted and ready to apply.
    best_models[name] = search.best_estimator_
    print(f"  Best ROC-AUC (CV): {search.best_score_:.3f}")
    print(f"  Best params:       {search.best_params_}\n")

print("Two-stage-selection models tuned and stored in `best_models`.")

# --- Matthews Correlation Coefficient (MCC) ---------------------------------
# MCC is included alongside ROC-AUC because it accounts for all four cells of
# the confusion matrix and remains informative under class imbalance — unlike
# accuracy or F1, it returns a low score when a model simply predicts the
# majority class for every sample. Values range from -1 (perfect
# misclassification) through 0 (no better than chance) to +1 (perfect).
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef

print("\n=== Cross-validated MCC (training set) ===\n")
for name, model in best_models.items():
    y_cv_pred = cross_val_predict(model, X_train, y_train, cv=cv)
    cv_mcc = matthews_corrcoef(y_train, y_cv_pred)
    print(f"  {name:25s}: MCC = {cv_mcc:.3f}")



# %%
# =============================================================================
# STEP 6: FINAL EVALUATION ON THE HELD-OUT TEST SET  — RUN ONCE
# =============================================================================
#
# This is the moment the entire design exists to protect. These 38 samples have
# played no part in any decision: not feature selection, not fitting, not
# tuning. The resulting metrics therefore constitute an honest estimate of
# performance on unseen data.
#
# CRITICAL DISCIPLINE: these results must not be inspected, followed by model
# adjustment, and then re-evaluated. Doing so converts the test set into part
# of the tuning process and silently inflates all metrics. The test set is
# evaluated once, and the outcome is reported as obtained.
#
# These scores are expected to be LOWER than the cross-validated estimates.
# CV estimates are mildly optimistic, and with n = 38 there is substantial
# sampling noise. That gap is normal and should be reported transparently
# rather than concealed.

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, matthews_corrcoef
)

print("=== FINAL TEST-SET PERFORMANCE (unseen data) ===\n")

final_results = []
for name, model in best_models.items():
    y_pred  = model.predict(X_test)                 # hard 0/1 predictions
    y_proba = model.predict_proba(X_test)[:, 1]     # P(high-risk), needed for AUC

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    final_results.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC_AUC": auc, "MCC": mcc})

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.3f}   F1: {f1:.3f}   ROC-AUC: {auc:.3f}   MCC: {mcc:.3f}")

   # INTERPRETING THE CONFUSION MATRIX — this is the clinically meaningful
    # output:
    #     [[TN  FP]      TN = low-risk correctly identified   <- the critical cell
    #      [FN  TP]]     TP = high-risk correctly identified
    #
    # The top-left cell is the most informative. A model with high accuracy but
    # TN = 0 has classified every sample as high-risk: it provides no
    # stratification and is clinically worthless despite appearing performant
    # on aggregate metrics.
    print("Confusion matrix  [rows=actual, cols=predicted]  (0=low, 1=high):")
    print(confusion_matrix(y_test, y_pred))

    # zero_division=0 prevents a crash when a model never predicts low-risk
    # (precision would otherwise be 0/0). The fact that this guard is needed
    # is itself a diagnostic signal.
    print(classification_report(y_test, y_pred, target_names=["low-risk", "high-risk"], zero_division=0))
    print()

final_df = pd.DataFrame(final_results).set_index("Model")
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print("=== Summary: final test-set metrics ===\n")
print(final_df)


# %%
# =============================================================================
# STEP 7: FIGURES FOR THE DISSERTATION (colourblind-safe)
# =============================================================================
#
# Generates five figures and exports two tables to results/. All figures use the
# Okabe–Ito colour-universal palette with redundant encoding (line style and
# hatching as well as hue), ensuring readability under colour vision deficiency
# and in greyscale print.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# savefig.dpi=300 is print quality; figure.dpi=150 keeps on-screen preview
# responsive. Top/right spines are removed and gridlines softened — standard
# journal figure styling.
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

# --- Okabe–Ito qualitative palette (colour-vision-deficiency safe) ----------
# Reference: Okabe & Ito (2008), "Color Universal Design".
# Eight hues that remain distinguishable under deuteranopia, protanopia, and
# tritanopia. This is a qualitative (unordered) palette — appropriate here
# because the four models are categories, not points on a scale. A sequential
# ramp such as viridis would imply a ranking that does not exist.
OKABE_ITO = {
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "bluegreen":  "#009E73",
    "purple":     "#CC79A7",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "yellow":     "#F0E442",
    "black":      "#000000",
}

COLOURS = {
    "Logistic Regression": OKABE_ITO["blue"],
    "Random Forest":       OKABE_ITO["vermillion"],
    "SVM":                 OKABE_ITO["bluegreen"],
    "XGBoost":             OKABE_ITO["purple"],
}

# Redundant encoding: colour alone should never be the sole channel carrying
# information. Line styles serve this purpose here; hatching is applied to the
# bar charts below.
LINESTYLES = {
    "Logistic Regression": "-",
    "Random Forest":       "--",
    "SVM":                 "-.",
    "XGBoost":             (0, (3, 1, 1, 1)),
}

# HARDCODED from the Step 5 run. Safe because the pipeline is seed-pinned and
# therefore deterministic — but if the seed, variance threshold, or pipeline
# structure is changed, Step 5 must be re-run and these values updated.
tuned_cv_auc = {"Logistic Regression": 0.634, "Random Forest": 0.600,
                "SVM": 0.641, "XGBoost": 0.621}


# --- FIGURE 1: ROC curves ---------------------------------------------------
# Plots true positive rate against false positive rate across every possible
# decision threshold. The diagonal represents chance. A curve hugging the
# diagonal indicates the model cannot separate the classes at ANY threshold —
# not merely that the default threshold is suboptimal.
fig, ax = plt.subplots(figsize=(6, 5.5))
for name, model in best_models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, color=COLOURS[name], ls=LINESTYLES[name], lw=2,
            label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

ax.plot([0, 1], [0, 1], color="black", ls=":", lw=1.4, label="Chance (AUC = 0.500)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on held-out test set (Discovery cohort, n = 38)")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.savefig(RESULTS_DIR / "fig1_roc_curves.png"); plt.close(fig)


# --- FIGURE 2: Confusion matrices -------------------------------------------
# 'cividis' is a sequential colourmap (appropriate here, as the values are
# proportions — an ordered quantity) engineered to be perceptually uniform for
# both normal colour vision and deuteranopia. Dark tones correspond to low
# values and bright yellow to high values, so the text colour condition is
# inverted relative to a conventional 'Blues' map.
fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
for ax, (name, model) in zip(axes.ravel(), best_models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))

    # Row-normalise for COLOUR only: each actual class is shaded by the
    # proportion predicted each way, preventing the larger high-risk row from
    # dominating visually. Raw counts remain in the cell text.
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_pct, cmap="cividis", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=15,
                    color="black" if cm_pct[i, j] > 0.5 else "white")

    ax.set_xticks([0, 1], ["Low-risk", "High-risk"])
    ax.set_yticks([0, 1], ["Low-risk", "High-risk"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(name, fontsize=11)
    ax.grid(False)   # gridlines over an image are visual noise

fig.suptitle("Confusion matrices on held-out test set (n = 38)", fontsize=12)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fig2_confusion_matrices.png"); plt.close(fig)


# --- FIGURE 3: Low-risk recall vs accuracy ----------------------------------
# This figure makes the imbalance problem visible: the accuracy bars appear
# respectable while the low-risk recall bars collapse. Random Forest is the
# clearest example — 0.71 accuracy, 0.00 recall. It identified none of the
# 10 low-risk individuals.
fig, ax = plt.subplots(figsize=(6.5, 4.5))
names = list(best_models.keys())
recalls, accs = [], []
for name in names:
    cm = confusion_matrix(y_test, best_models[name].predict(X_test))
    recalls.append(cm[0, 0] / cm[0].sum())          # TN / all actual low-risk
    accs.append((cm[0, 0] + cm[1, 1]) / cm.sum())   # (TN + TP) / total

x = np.arange(len(names)); w = 0.38   # w = bar width; x ± w/2 offsets the pair
ax.bar(x - w/2, accs, w, label="Overall accuracy",
       color=OKABE_ITO["skyblue"], edgecolor="black", linewidth=0.6)
ax.bar(x + w/2, recalls, w, label="Low-risk recall",
       color=OKABE_ITO["vermillion"], edgecolor="black", linewidth=0.6, hatch="//")
for i, (a, r) in enumerate(zip(accs, recalls)):
    ax.text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)

ax.set_xticks(x, names, rotation=20, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
ax.set_title("Accuracy conceals failure on the minority class")
ax.legend(frameon=False)
fig.savefig(RESULTS_DIR / "fig3_low_risk_recall.png"); plt.close(fig)


# --- FIGURE 4: Cross-validated vs held-out AUC ------------------------------
# Illustrates the optimism gap. CV scores are computed on data the tuning
# process searched over and are therefore mildly inflated; the test bars
# represent the honest estimate. The chance line makes explicit how close the
# honest estimates sit to random guessing.
fig, ax = plt.subplots(figsize=(6.5, 4.5))
cv_vals = [tuned_cv_auc[n] for n in names]
test_vals = [final_df.loc[n, "ROC_AUC"] for n in names]

ax.bar(x - w/2, cv_vals, w, label="Cross-validated (training)",
       color=OKABE_ITO["skyblue"], edgecolor="black", linewidth=0.6)
ax.bar(x + w/2, test_vals, w, label="Held-out test",
       color=OKABE_ITO["blue"], edgecolor="black", linewidth=0.6, hatch="\\\\")
ax.axhline(0.5, color="black", ls=":", lw=1.4, label="Chance")
ax.set_xticks(x, names, rotation=20, ha="right")
ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 0.8)
ax.set_title("Cross-validated vs held-out performance")
ax.legend(frameon=False, fontsize=9)
fig.savefig(RESULTS_DIR / "fig4_cv_vs_test.png"); plt.close(fig)


# --- FIGURE 5: The confound effect — THE KEY FIGURE -------------------------
# The central result of the dissertation. Compares test AUC when trained on all
# 333 samples (platform confounded with outcome) against Discovery-only
# (confound removed). Every model drops by 0.16–0.25 AUC, to near chance.
#
# The interpretation: the apparent predictive performance on the full dataset
# derived substantially from array-version signal, not methylation biology.
fig, ax = plt.subplots(figsize=(6.5, 4.5))

# HARDCODED from the earlier full-dataset run, which is no longer reproduced
# by this script (Step 3 now filters to Discovery). Retained here so the
# comparison can be drawn. Source: Step 6 output prior to the Discovery
# restriction.
confounded = {"Logistic Regression": 0.744, "Random Forest": 0.721,
              "SVM": 0.779, "XGBoost": 0.777}

before = [confounded[n] for n in names]
after  = [final_df.loc[n, "ROC_AUC"] for n in names]

ax.bar(x - w/2, before, w, label="All 333 samples (confounded)",
       color=OKABE_ITO["vermillion"], edgecolor="black", linewidth=0.6)
ax.bar(x + w/2, after,  w, label="Discovery only (confound-free)",
       color=OKABE_ITO["bluegreen"], edgecolor="black", linewidth=0.6, hatch="//")
ax.axhline(0.5, color="black", ls=":", lw=1.4, label="Chance")

# Arrows from each 'before' bar to its 'after' bar, making the magnitude of
# the drop explicit rather than requiring the reader to compare bar heights.
for i, (b, a) in enumerate(zip(before, after)):
    ax.annotate("", xy=(i + w/2, a), xytext=(i - w/2, b),
                arrowprops=dict(arrowstyle="->", color="dimgrey", lw=1.4))

ax.set_xticks(x, names, rotation=20, ha="right")
ax.set_ylabel("Test-set ROC-AUC"); ax.set_ylim(0, 0.9)
ax.set_title("Removing the platform confound collapses performance to chance")
ax.legend(frameon=False, fontsize=9)
fig.savefig(RESULTS_DIR / "fig5_confound_effect.png"); plt.close(fig)


# --- Export tables ----------------------------------------------------------
# CSV format allows the exact values to be incorporated into the write-up
# without transcription errors, and the files regenerate automatically if
# anything upstream changes.
results_df.to_csv(RESULTS_DIR / "table5_cv_comparison.csv")
final_df.to_csv(RESULTS_DIR / "table7_final_test.csv")

print(f"Figures and tables written to:\n{RESULTS_DIR}") 

