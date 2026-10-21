# =============================================================================
# 01_PREPARE_DATA.PY
#
# Purpose: turn the raw GEO download (GSE315764) into a clean, model-ready
#          dataset cached on disk.
#
# This script only needs to be run once. Everything it produces is saved to
# Data/interim/, so once it has run successfully, it does not need to be run
# again
# The modelling script reads the outputs from this code.
# Outputs:
#   Data/interim/sample_labels.csv             <- sample metadata including cohort, platform, and risk classification
#   Data/interim/methylation_filtered.parquet  <- processed methylation matrix for downstream modelling
# =============================================================================

import gzip
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# --- Folder locations -------------------------------------------------------
# Defined once here and referenced by all subsequent steps to ensure consistency and maintainability across the pipeline

# The repository root (the folder above code/), so the pipeline runs wherever
# the repository is cloned.
BASE_DIR    = Path(__file__).resolve().parents[1]
RAW_DIR    = BASE_DIR / "Data" / "raw"        # untouched downloads live here
INTERIM_DIR = BASE_DIR / "Data" / "interim"    # everything this script produces
CSV_PATH    = RAW_DIR / "GSE315764_combined_methylation.csv.gz"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)  # create interim/ if it's not there yet


# =============================================================================
# STEP 1: PARSE SERIES MATRICES, BUILD SAMPLE LABELS
# =============================================================================
#
# The methylation matrix contains beta values but no phenotypic annotations.
# Sample metadata, including risk cluster membership, is stored separately
# in the GEO series matrix files and must be extracted independently.
#
# The dataset spans two Illumina array platforms, each with its own series
# matrix file:
#   GPL21145 = Infinium MethylationEPIC v1
#   GPL33022 = Infinium MethylationEPIC v2

series_matrix_files = {
    "GPL21145": RAW_DIR / "GSE315764-GPL21145_series_matrix.txt.gz",
    "GPL33022": RAW_DIR / "GSE315764-GPL33022_series_matrix.txt.gz",
}


def parse_series_matrix(path, platform):
    """
    Pull the sample titles and GEO accession IDs out of one series matrix file.

    A series matrix is mostly metadata: each line starts with a '!' tag and
    holds one field for every sample, tab-separated. Only two lines are needed:

      !Sample_title          -> e.g. "C2_Discovery_1"  (encodes cluster + cohort)
      !Sample_geo_accession  -> e.g. "GSM9436409"      (GEO's own sample ID)

    The files are gzipped, but small — they are read directly without unzipping.
    """
    titles, accessions = [], []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Split on tabs, drop the leading '!Tag' element, strip the quotes
            # GEO wraps every value in.
            if line.startswith("!Sample_title"):
                titles = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                accessions = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]

    df = pd.DataFrame({"sample_title": titles, "gsm_id": accessions})
    df["platform"] = platform     # tag each sample with the array it was run on
    return df


# Parse both platform files and stack them into one table of 333 samples.
labels = pd.concat(
    [parse_series_matrix(path, plat) for plat, path in series_matrix_files.items()],
    ignore_index=True,
)

# The sample titles follow a self-describing naming convention:
# "C2_Discovery_1" encodes both the risk cluster (C2) and the cohort
# (Discovery). Splitting on underscores is therefore sufficient, avoiding
# the need to parse GEO's less structured characteristics fields.
labels["cluster"] = labels["sample_title"].str.split("_").str[0]   # C2 / C3 / C5 / C6
labels["cohort"]  = labels["sample_title"].str.split("_").str[1]   # Discovery / Replication

# Collapse the four clusters into the binary target variable.
# Cluster 2 represents the clinically low-risk group; clusters 3, 5, and 6
# correspond to high-risk metabolic profiles. (Note: cluster 4 is absent
# from this dataset.)
labels["risk"] = (labels["cluster"] != "C2").astype(int)   # 0 = low-risk, 1 = high-risk

# Save the labels independently. The modelling script joins these onto the
# methylation data, and having a small standalone file permits quick
# inspection without loading the 1.7 GB matrix.
labels.to_csv(INTERIM_DIR / "sample_labels.csv", index=False)


# --- Sanity checks ----------------------------------------------------------
# Each assertion below confirms a structural assumption on which the remainder
# of the pipeline depends. Expected: 333 samples, clusters C2/C3/C5/C6,
# 50 low-risk.

print(f"Total samples: {len(labels)}\n")
print("Samples per platform:\n", labels["platform"].value_counts(), "\n")
print("Samples per cohort:\n", labels["cohort"].value_counts(), "\n")
print("Samples per cluster:\n", labels["cluster"].value_counts(), "\n")
print("Class balance (0 = low-risk, 1 = high-risk):\n", labels["risk"].value_counts(), "\n")

# The two crosstabs below revealed the key structural problem in this dataset:
#
#   cohort x platform -> Discovery is 100% EPIC v1, Replication is 100% v2.
#                        Cohort and platform are perfectly confounded.
#   cohort x cluster  -> ALL 50 low-risk samples are in Discovery;
#                        Replication is 100% high-risk.
#
# Together these mean array version predicts the label perfectly: every v2
# sample is high-risk. A model trained on all 333 samples could score well by
# detecting the array's technical signature rather than any biology. This confound is
# why the modelling script restricts the analysis to the Discovery cohort.
print("Cohort x platform:\n", pd.crosstab(labels["cohort"], labels["platform"]), "\n")
print("Cohort x cluster:\n", pd.crosstab(labels["cohort"], labels["cluster"], margins=True), "\n")
print("Risk proportions within each cohort:\n",
      pd.crosstab(labels["cohort"], labels["risk"], normalize="index"), "\n")

print(labels.head())


# =============================================================================
# STEP 1b: INSPECT THE METHYLATION CSV'S STRUCTURE
# =============================================================================
#
# Before committing to a 1.7 GB load, a quick read of the first five rows
# confirms the assumed file layout. pandas decompresses the .gz on the fly
# and stops early, so this check is inexpensive.
#
# Confirmed by this inspection:
#   - Shape (5, 334)     -> one CG_id column + 333 sample columns
#   - Columns are named by SAMPLE TITLE (not GSM id) -> determines the join key
#   - Values fall between 0 and 1 -> these are processed beta values, so no raw
#     IDAT processing is required (and GSE315764_RAW.tar was therefore not used)

preview = pd.read_csv(CSV_PATH, nrows=5)

print("Preview shape (rows, columns):", preview.shape)
print("\nName of the first column:", repr(preview.columns[0]))
print("\nFirst 8 column names:")
print(list(preview.columns[:8]))
print("\nLast 3 column names:")
print(list(preview.columns[-3:]))
print("\nA corner of the actual data (first 5 rows, first 5 columns):")
print(preview.iloc[:, :5])


# =============================================================================
# STEP 2: LOAD METHYLATION MATRIX, VARIANCE-FILTER, JOIN LABELS
# =============================================================================
#
# This section performs the most resource-intensive processing. Its purpose is
# to convert one large, column-oriented CSV into a compact Parquet file that
# the modelling script can load in seconds.
 
# --- 1. Load the full matrix ------------------------------------------------
# The file stores CpG sites as rows and samples as columns, indexed by CG_id
# (e.g. "cg25335775"). This is the slow, memory-intensive step: it requires
# several minutes and approximately 2 GB of RAM. It is executed only once.
print("Loading CSV (this takes a few minutes)...")
meth = pd.read_csv(CSV_PATH, index_col="CG_id")
print("Loaded raw matrix:", meth.shape, "(CpGs x samples)")

# --- 2. Transpose -----------------------------------------------------------
# scikit-learn expects one row per sample and one column per feature, which is
# the inverse of the file's storage layout. The matrix is therefore transposed.
X = meth.T
del meth   # release the original immediately to conserve memory
print("Transposed:", X.shape, "(samples x CpGs)")

# --- 3. Variance filtering --------------------------------------------------
# The array measures ~677k CpG sites, many of which are essentially constant
# across all participants. Such probes carry no discriminative information and
# represent only noise and memory overhead.
#
# This filter serves a dual purpose: it constitutes the first stage of the
# feature-selection strategy and reduces the dataset to a manageable size.
# The threshold is intentionally lenient — it removes only genuinely invariant
# probes. More selective feature reduction (via LASSO and Random Forest) is
# applied later, within cross-validation, to prevent information leakage from
# test folds into feature selection.
selector = VarianceThreshold(threshold=0.001)
X_var = selector.fit_transform(X)
kept_cpgs = X.columns[selector.get_support()]
X = pd.DataFrame(X_var, index=X.index, columns=kept_cpgs)
print("After variance filter:", X.shape, "(samples x CpGs kept)")

# --- 4. Attach the labels ---------------------------------------------------
# The CSV's column headers are sample titles, which now form X's row index.
# These join directly onto the labels table from Step 1. Cluster, cohort,
# and platform are retained alongside the target variable: although they are
# not model inputs, the modelling script requires 'cohort' to isolate the
# Discovery samples.
labels = pd.read_csv(INTERIM_DIR / "sample_labels.csv").set_index("sample_title")
data = X.join(labels[["risk", "cluster", "cohort", "platform"]], how="inner")

# --- 5. Integrity check -----------------------------------------------------
# An inner join silently drops unmatched rows, meaning a naming mismatch
# between the two source files would reduce the dataset without raising an
# error. The following diagnostics ensure any such discrepancy is immediately
# apparent.
print("\nRows after join:", len(data), "(expected 333)")
print("Any unmatched samples:", len(X) - len(data))
print("Class balance after join:\n", data["risk"].value_counts())

# --- 6. Cache to Parquet ----------------------------------------------------
# Parquet is a compressed columnar format that preserves dtypes, loads
# substantially faster than CSV, and occupies a fraction of the disk space.
# Writing the result here converts the 1.7 GB load from a per-run cost into
# a one-time operation.
out_path = INTERIM_DIR / "methylation_filtered.parquet"
data.to_parquet(out_path)
print(f"\nSaved cached dataset to:\n{out_path}") 

