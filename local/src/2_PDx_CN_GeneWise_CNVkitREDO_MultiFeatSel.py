#!/usr/bin/env python
# coding: utf-8

# Compute CNV stats for intogen genes that map to PDx segmented CN data (from CNVkit)

# ### Imports
# Import libraries and write settings here.
import numpy as np
import pandas as pd
from helpers import remove_collinear_features
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_curve, multilabel_confusion_matrix, auc, matthews_corrcoef, roc_auc_score, accuracy_score
# feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif

import pickle
from helpers import combine_binary_features

import matplotlib.pyplot as plt
import seaborn as sb
# Set default font size
sb.set(font_scale=1.2)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)


# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

logfile = snakemake.log[0]
target_col = snakemake.params.target_col
# load sample id conversion table, drug response data
drug_response_data = pd.read_csv(snakemake.input.response,
                                 sep="\t")
ctx3w_cat = drug_response_data[["ircc_id", target_col]].\
    set_index("ircc_id")

# parse PDx segmented CNV data
PDx_CNV_data = pd.read_csv(snakemake.input.cnv,
                           sep="\t", index_col=None,
                           names=["chr",
                                  "begin",
                                  "end",
                                  "sample",
                                  "log2R",
                                  "seg_CN",
                                  "depth",
                                  "p_ttest",
                                  "probes",
                                  "weight",
                                  "gene_chr",
                                  "gene_b",
                                  "gene_e",
                                  "gene_symbol",
                                  'tumor_types',
                                  "overlapping_admire_segs",
                                  "length_segment-gene_overlap"])
PDx_CNV_data["seg_id"] = PDx_CNV_data.agg(
    lambda x: f"{x['chr']}:{x['begin']}-{x['end']};{x['sample']}", axis=1)
PDx_CNV_data["gene_HUGO_id"] = PDx_CNV_data["gene_symbol"]
# drop segments missing gene symbol
PDx_CNV_data = PDx_CNV_data[PDx_CNV_data["gene_HUGO_id"] != "."]
PDx_CNV_data["sample"] = PDx_CNV_data["sample"].apply(lambda x: x+"_hum")
# load shared set of genes for intogen and targeted sequencing
common_geneset = pd.read_table(snakemake.input.targeted,
                               header=None, sep="\t")
common_geneset = common_geneset[0].tolist()

# merge with genealogy using Sanger ids
PDx_CNV_data = pd.merge(PDx_CNV_data,
                        drug_response_data[["sanger_id", "sample_level",
                                            "ircc_id", "is_LMX", "ircc_id_short",
                                            "passage", 'lineage']].drop_duplicates(),
                        left_on="sample",
                        right_on="sanger_id")

# reshape into a gene x model CNV matrix
in_df = PDx_CNV_data[["gene_HUGO_id", "ircc_id", "log2R"]]
# merge multiple CNV values for same gene;model
in_df = PDx_CNV_data.groupby(["gene_HUGO_id", "ircc_id"]).\
    agg({"log2R": "mean"}).reset_index()

# bin log2R into gain/neutral/loss according to gistic thresholds
loss_thr = -.2
gain_thr = .1
min_CN = round(in_df.log2R.min()) - 1
max_CN = round(in_df.log2R.max()) + 1
in_df["gene_direction"] = pd.cut(in_df.log2R,
                                 bins=[min_CN, loss_thr,
                                       gain_thr, max_CN],
                                 labels=["Loss", "Neutral", "Gain"]).astype(str)
# drop neutral calls
in_df = in_df[~in_df["gene_direction"].str.contains("Neutral")]

# load TCGA CNV gene stats
TCGA_CNV_gene_data = pd.read_csv(snakemake.input.TCGA_CNV,
                                 sep="\t", header=0)
TCGA_CNV_gene_data["event_source"] = TCGA_CNV_gene_data.event_id.\
    apply(lambda x: "gistic2" if (
        "Deletion" in x or "Amplification" in x) else "admire1.2")
# merge TCGA & PDx CNV data, this forces TCGA's CNV direction for e/a gene
in_df = pd.merge(in_df,
                 TCGA_CNV_gene_data.dropna(),
                 left_on=["gene_HUGO_id", "gene_direction"],
                 right_on=["HUGO_id", "event_direction"])[
                     ["gene_HUGO_id", "ircc_id", "gene_direction"]].\
    drop_duplicates().sort_values("ircc_id").set_index("ircc_id")

# encode gene and direction as binary features,
# account for multiple CNV events for each sample
in_df = in_df.gene_HUGO_id.astype(str) + "_" + in_df.gene_direction.astype(str)
CNV_matrix = pd.get_dummies(in_df).reset_index().groupby("ircc_id").sum()

# load drug response data
ctx3w_cat = drug_response_data[["ircc_id", target_col]].\
    set_index("ircc_id")
features_in = pd.merge(ctx3w_cat, CNV_matrix,
                       right_index=True, left_index=True)
# replace na w/t 0
features_in = features_in.fillna(0)
# drop instances w/t missing target
features_in = features_in[~features_in[target_col].isna()].\
    drop_duplicates()

# train-test split
features_col = [c for c in features_in.columns if c != target_col]
TT_df = drug_response_data[drug_response_data.ircc_id.isin(features_in.index)][
    ["ircc_id", "is_test"]]
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()
X_train = features_in.loc[train_models, features_col]
y_train = features_in.loc[train_models, target_col]
X_test = features_in.loc[test_models, features_col]
y_test = features_in.loc[test_models, target_col]

# remove features with low variance i.e. where most values are 1 or 0
var_thrs = snakemake.params.var_thrs
var_thrs_float = float(var_thrs[:-1])/100
thresholder = VarianceThreshold(threshold=(
    var_thrs_float * (1 - var_thrs_float)))
features_tokeep = X_train.columns[thresholder.fit(X_train).get_support()]
X_train_clean = X_train[features_tokeep]
X_test_clean = X_test[features_tokeep]

# combine similar features via product
similarity_trsh = snakemake.params.similarity_trsh
X_train_clean_combine = combine_binary_features(X_train_clean,
                                                similarity_trsh,
                                                max_combine=3)
# transform X test adding the combined features
combined_features = [c.split("+")
                     for c in X_train_clean_combine.columns if "+" in c]
X_test_clean_combine = X_test_clean.copy()
for cols in combined_features:
    X_test_clean_combine["+".join(cols)] = X_test_clean[cols].product(axis=1)
    X_test_clean_combine.drop(cols, axis=1, inplace=True)

# univariate feature selection via ANOVA f-value filter
ANOVA_pctl = snakemake.params.ANOVA_pctl
ANOVA_support = SelectPercentile(f_classif,
                                 percentile=ANOVA_pctl).\
    fit(X_train_clean_combine, y_train).get_support()
ANOVA_selected = X_train_clean_combine.columns[ANOVA_support]

# transform train, test datasets
X_train = X_train_clean_combine[ANOVA_selected]
X_test = X_test_clean_combine[ANOVA_selected]
# save X datasets
X_train.to_csv(snakemake.output.Xtrain, sep="\t")
X_test.to_csv(snakemake.output.Xtest, sep="\t")

X_train = X_train.values
Y_train = y_train.values
X_test = X_test.values
Y_test = y_test.values

# train linearSVM
svm = LinearSVC(penalty="l1", dual=False, max_iter=5000).fit(X_train, Y_train)

# calc performance metrics
y_classes = features_in[target_col].unique().tolist()
Y_pred = svm.predict(X_test)
multi_cm = multilabel_confusion_matrix(Y_test, Y_pred, labels=y_classes)
tn, fp, fn, tp = [i for i in sum(multi_cm).ravel()]
accuracy = tp + tn / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
# harmonic mean of precion and recall
F1 = 2*(precision * recall) / (precision + recall)
model_mcc = matthews_corrcoef(Y_test, Y_pred)

with open(snakemake.log[0], "a") as logfile:
    printout = f"Model: LinearSVC | n_features: {ANOVA_selected.shape} |\
         Precision: {precision:.4f} |\
         Recall: {recall:.4f} | MCC: {model_mcc:.4f} | \
            F1: {F1:.4f} | Accu: {accuracy:.4f}"
    logfile.write(printout)

# get linear SVC feature coefficients
coeff_plot_df = pd.DataFrame(svm.coef_.T,
                             columns=svm.classes_,
                             index=ANOVA_selected)
coeff_plot_df = coeff_plot_df.stack().reset_index()
coeff_plot_df.columns = ["feature", "class", "coeff"]
coeff_plot_df = coeff_plot_df.sort_values("coeff")
# select top / bottom features
top = pd.concat(
    [coeff_plot_df.head(15), coeff_plot_df.tail(15)]).feature.unique()
plot_df = coeff_plot_df[coeff_plot_df.feature.isin(top)]

fig, ax = plt.subplots(figsize=(10, 16))
ax = sb.barplot(x="coeff",
                y="feature",
                hue="class",
                hue_order=["PD", "SD", "OR"],
                palette="Set2",
                data=plot_df)
st = fig.suptitle(printout, y=.95, fontsize=18)
fig.tight_layout
fig.savefig(snakemake.output.loadings_fig,
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "snakemake CN_FeatClean"})
