#!/usr/bin/env python
# coding: utf-8

# Data manipulation
import pandas as pd
import numpy as np

from helpers import remove_collinear_features

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import multilabel_confusion_matrix, matthews_corrcoef, confusion_matrix, accuracy_score
# feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, chi2

from helpers import combine_binary_features

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Set default font size
plt.rcParams['font.size'] = 24
# Set default font size
sb.set(font_scale=.8)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)

logfile = snakemake.log[0]
# load sample id conversion table, drug response data
drug_response_data = pd.read_csv(snakemake.input.response, sep="\t")

# load driver annotation for PDx models
driver_data = pd.read_csv(snakemake.input.mut,
                          "\t", header=0).rename(columns={'Sample': 'sanger_id'})

# + driver annot data
drug_mut_df = pd.merge(drug_response_data, driver_data,
                       on="sanger_id", how="left")

# transform df to have a gene x sample binary mutation matrix
# including all driver genes
target_col = snakemake.params.target_col
features_pre = drug_mut_df[drug_mut_df["Final_Driver_Annotation"] == True][
    ["ircc_id",
     "Gene",
     target_col]
].drop_duplicates()
# 1-hot encode genes, vector sum on sample to
# account for multiple mut in same sample
features_in = pd.get_dummies(features_pre.Gene)
features_in["ircc_id"] = features_pre.ircc_id
features_in = features_in.groupby("ircc_id").sum()

# add drug response as target
df1 = features_pre[[target_col, "ircc_id"]
                   ].drop_duplicates().set_index("ircc_id")
features_in = pd.concat([features_in, df1], axis=1)

# replace na w/t 0
features_in = features_in.fillna(0)
# drop instances w/t missing target
features_in = features_in[~features_in[target_col].isna()].\
    drop_duplicates()

# clean features
features_clean = features_in.drop(target_col, axis=1)

# add some known driver mutation combos for CRC
features_clean["KRAS-BRAF-NRAS_triple_neg"] = features_in[["KRAS",
                                                           "BRAF",
                                                           "NRAS"]].sum(axis=1).\
    replace({0: 1, 1: 0, 2: 0, 3: 0})
features_clean["KRAS-APC_double_pos"] = features_in[["KRAS", "APC"]].sum(axis=1).\
    replace({1: 0, 2: 1})
features_col = features_clean.columns
print(len(features_col))
features_clean[target_col] = features_in[target_col]

TT_df = drug_response_data[drug_response_data.ircc_id.isin(features_clean.index)][
    ["ircc_id", "is_test"]]
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()

# train-test split
X_train = features_clean.loc[train_models, features_col]
y_train = features_clean.loc[train_models, target_col]
X_test = features_clean.loc[test_models, features_col]
y_test = features_clean.loc[test_models, target_col]

# remove features with low variance i.e. where most values are 1 or 0
var_thrs = snakemake.params.var_thrs
var_thrs_float = float(var_thrs[:-1])/100
thresholder = VarianceThreshold(threshold=(
    var_thrs_float * (1 - var_thrs_float)))
features_tokeep = X_train.columns[thresholder.fit(X_train).get_support()]
X_train = X_train[features_tokeep]
X_test = X_test[features_tokeep]
print(len(features_tokeep))
# combine similar features via product
similarity_trsh = snakemake.params.similarity_trsh
X_train = combine_binary_features(X_train, similarity_trsh, max_combine=3)
# transform X test adding the combined features
combined_features = [c.split("+")
                     for c in X_train.columns if "+" in c]
X_test_clean_combine = X_test.copy()
for cols in combined_features:
    X_test_clean_combine["+".join(cols)] = X_test[cols].product(axis=1)
    X_test_clean_combine.drop(cols, axis=1, inplace=True)
X_test = X_test_clean_combine
print(len(X_test_clean_combine.columns))
# univariate feature selection via ANOVA f-value filter
ANOVA_pctl = snakemake.params.ANOVA_pctl
ANOVA_support = SelectPercentile(chi2,
                                 percentile=ANOVA_pctl).\
    fit(X_train, y_train).get_support()
ANOVA_selected = X_train.columns[ANOVA_support]
print(len(ANOVA_selected))

# transform train, test datasets
X_train = X_train[ANOVA_selected]
X_test = X_test[ANOVA_selected]
# save to file
X_train.to_csv(snakemake.output.Xtrain, sep="\t")
X_test.to_csv(snakemake.output.Xtest, sep="\t")

X_train = X_train.values
Y_train = y_train.values
X_test = X_test.values
Y_test = y_test.values

# train linearSVM
svm = LinearSVC(penalty="l1", dual=False, max_iter=5000).fit(X_train, Y_train)

# calc performance metric
y_classes = features_clean[target_col].unique().tolist()
Y_pred = svm.predict(X_test)

# if multiclass predictor
if len(svm.classes_) > 2:
    cm = multilabel_confusion_matrix(Y_test, Y_pred, labels=y_classes)
    tn, fp, fn, tp = [i for i in sum(cm).ravel()]
else:
    cm = confusion_matrix(Y_test, Y_pred)
    tn, fp, fn, tp = cm.ravel()

accuracy = tp + tn / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
# harmonic mean of precion and recall
F1 = 2*(precision * recall) / (precision + recall)
model_mcc = matthews_corrcoef(Y_test, Y_pred)
printout = f"Model: LinearSVC |\
     n_features: {ANOVA_selected.shape[0]} | \
     Precision: {precision: .4f} | \
     Recall: {recall: .4f} | \
     MCC: {model_mcc: .4f} | \
     F1: {F1: .4f} | \
     Accu: {accuracy: .4f}"

# if multiclass
if len(svm.classes_) > 2:
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
                    hue_order=sorted(y_classes),
                    palette="Set2",
                    data=plot_df)
else:
    # get linear SVC feature coefficients
    coeff_plot_df = pd.DataFrame(svm.coef_.T,
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
                    palette="Set2",
                    data=plot_df)
st = fig.suptitle(printout, y=.95, fontsize=18)
fig.tight_layout
fig.savefig(snakemake.output.loadings_fig,
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "snakemake mut_FeatClean"})
