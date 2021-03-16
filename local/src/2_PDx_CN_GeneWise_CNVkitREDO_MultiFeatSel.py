#!/usr/bin/env python
# coding: utf-8

# Compute CNV stats for intogen genes that map to PDx segmented CN data (from CNVkit)

# ### Imports
# Import libraries and write settings here.
import numpy as np
import pandas as pd
from helpers import remove_collinear_features
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_curve, multilabel_confusion_matrix, auc, matthews_corrcoef, roc_auc_score, accuracy_score

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
target_col = "Cetuximab_Standard_3wks_cat"
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
# merge multiple CNV values for same gene;model (TODO: check these)
in_df = PDx_CNV_data.groupby(["gene_HUGO_id", "ircc_id"]).agg({
    "log2R": "mean"}).reset_index()
CNV_matrix = in_df.set_index(["ircc_id", "gene_HUGO_id"]).unstack()
CNV_matrix.columns = CNV_matrix.columns.get_level_values(
    "gene_HUGO_id").tolist()
CNV_matrix.index = CNV_matrix.index.get_level_values("ircc_id").tolist()

ctx3w_gat = drug_response_data[[
    "ircc_id", target_col]].set_index("ircc_id")
features_in = pd.merge(ctx3w_cat, CNV_matrix,
                       right_index=True, left_index=True)

common_geneset = common_geneset
genes_tokeep = [g for g in features_in.columns if g in set(common_geneset)]
features_clean = features_in
features_col = np.array([c for c in features_clean.columns if c != target_col])
features_clean = features_clean[features_col]

# remove features with low variance
var_trsh = features_clean.var(axis=0).\
    describe().loc[snakemake.params.var_pctl]
features_clean = features_clean[(features_clean.var(axis=0) > var_trsh).index]

# remove colinear features
features_clean = remove_collinear_features(
    features_clean,
    snakemake.params.colinear_trsh,
    priority_features=genes_tokeep,
    logfile=logfile)
features_col = features_clean.columns

# replace na w/t 0
features_clean = features_clean.fillna(0)
features_clean[target_col] = features_in[target_col]

# drop instances w/t missing target
features_clean = features_clean[~features_clean[target_col].isna()].\
    drop_duplicates()

TT_df = drug_response_data[drug_response_data.ircc_id.isin(features_clean.index)][
    ["ircc_id", "is_test"]]
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()

# train-test split
X_train = features_clean.loc[train_models, features_col]
y_train = features_clean.loc[train_models, target_col]
X_test = features_clean.loc[test_models, features_col]
y_test = features_clean.loc[test_models, target_col]

# standardise features
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train.values),
                       columns=features_col,
                       index=train_models)
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test.values),
                      columns=features_col,
                      index=test_models)
# save X datasets
X_train.to_csv(snakemake.output.Xtrain, sep="\t")
X_test.to_csv(snakemake.output.Xtest, sep="\t")

X_train = X_train.values
Y_train = y_train.values
X_test = X_test.values
Y_test = y_test.values

# train linearSVM
svm = LinearSVC().fit(X_train, Y_train)

y_classes = features_clean[target_col].unique().tolist()
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
    printout = f"Model: LinearSVC | Precision: {precision:.4f} |\
        Recall: {recall:.4f} | MCC: {model_mcc:.4f} | \
            F1: {F1:.4f} | Accu: {accuracy:.4f}"
    logfile.write(printout)

# get linear SVC feature coefficients
coeff_plot_df = pd.DataFrame(svm.coef_.T,
                             columns=svm.classes_,
                             index=features_col)
coeff_plot_df = coeff_plot_df.stack().reset_index()
coeff_plot_df.columns = ["feature", "class", "coeff"]
coeff_plot_df = coeff_plot_df.sort_values("coeff")
# select top / bottom features
top = pd.concat(
    [coeff_plot_df.head(10), coeff_plot_df.tail(10)]).feature.unique()
plot_df = coeff_plot_df[coeff_plot_df.feature.isin(top)]

fig, ax = plt.subplots(figsize=(10, 16))
ax = sb.barplot(x="coeff",
                y="feature",
                hue="class",
                palette="Set2",
                data=plot_df)
st = fig.suptitle(printout, y=.95, fontsize=18)
fig.tight_layout
fig.savefig(snakemake.output.loadings_fig,
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "snakemake CN_FeatClean"})
