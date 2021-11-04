#!/usr/bin/env python
# coding: utf-8

# Introduction

# Imports
import sys
# Data manipulation
import pandas as pd
import numpy as np

# feature selection
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2


# Load input files

# load sample id conversion table, drug response data
f = snakemake.input.response
#f = "../../../dataset/4_McNemar_PDX/DrugResponse_LMXfirslevel_trainTest8.tsv"
drug_response_data = pd.read_csv(f, sep="\t")
# load driver annotation for PDx models
f = snakemake.input.mut
#f = "../../../local/share/data//Driver_Annotation/CodingVariants_All669PDX_samples_26Feb2020_annotated_drivers_shortVersionForPDXfinder_EK.txt"
driver_data = pd.read_csv(f,
                          "\t", header=0).rename(
                              columns={'Sample': 'sanger_id'})
# + driver annot data
drug_mut_df = pd.merge(drug_response_data, driver_data,
                       on="sanger_id", how="left")
# transform df to have a gene x sample binary mutation matrix
# including all driver genes
features_pre = drug_mut_df[drug_mut_df["Final_Driver_Annotation"] == True][
    ["ircc_id",
     "Gene"]
].drop_duplicates()

# Feature preprocessing

# 1-hot encode genes, vector sum on sample to
features_in = pd.get_dummies(features_pre.Gene)
features_in["ircc_id"] = features_pre.ircc_id
# account for multiple mut in same sample
features_in = features_in.groupby("ircc_id").sum()

# encode target variable to binary
#target_col = snakemake.params.target_col
target_col = "Cetuximab_Standard_3wks_cat"
Y = drug_response_data.set_index('ircc_id')
Y_class_dict = {'PD': 0, 'OR+SD': 1}
Y[target_col] = Y[target_col].replace(Y_class_dict)
# merge features and target, remove instances w/t missing target value
feature_col = features_in.columns
all_df = pd.merge(features_in, Y[target_col], right_index=True,
                  left_index=True, how="right")
all_df = all_df.dropna(axis=0, how='all')

# fillna in features with 0
all_df = all_df.fillna(0)
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]
# save 'raw' features aka no feature crosses
if snakemake.params.raw:
    all_df[feature_col].to_csv(snakemake.output.raw_mut,
                                   sep='\t')
    sys.exit()

# train-test split
train_models = Y[Y.is_test == False].index.unique()
test_models = Y[Y.is_test == True].index.unique()
X_train = all_df.loc[train_models, feature_col]
y_train = all_df.loc[train_models, target_col]
X_test = all_df.loc[test_models, feature_col]
y_test = all_df.loc[test_models, target_col]

# Feature engineering

# univariate chi2 w/t binary labels to filter top features for crosses
# chi2 stat is computed exclusively on the trainig set to prevent leakage
chi2_stat, pval = [pd.Series(arr) for arr in chi2(X_train, y_train)]
chi2_df = pd.concat([chi2_stat, pval], axis=1)
chi2_df.index = feature_col
chi2_df.columns = ['chi2_stat', 'Pval']
chi2_df = chi2_df.sort_values('chi2_stat', ascending=False)
# set a chi2 pctl threshold
pctl = snakemake.params.univ_ptl
pctl_tr = chi2_df.chi2_stat.describe().loc[pctl]
# get top gene features sorted by chi2 stat
top_features = chi2_df[(chi2_df.chi2_stat > pctl_tr)].index

# build new features counting the unique number of protein mut per gene per sample
pre = drug_mut_df[drug_mut_df["Final_Driver_Annotation"] == True][
    ["ircc_id",
     "Gene",
     "Protein"]
].drop_duplicates().groupby(["ircc_id", "Gene"]).Protein.nunique().\
    unstack()

# compute multiple mutation features
# these features are 0 when there's either 0 or 1 mut in gene, else
# they hold the count of unique (>1) mut per gene
multiple_mut = pre.replace({1: np.nan}).dropna(axis=1, how='all').fillna(0)
multiple_mut.columns = [c+"_MultiMut" for c in multiple_mut.columns]
# count tot unique mut burden per sample (aka sample mutational burden)
multiple_mut["unique_mut_burden"] = pre.apply(pd.Series.sum, axis=1)
multiple_mut.shape

# compute feature crosses (combinarions of 2,3 features)
# use only top gene features (by chi2 on binary labels),
# add feature crossses to the feature matrix
interactions2 = list(combinations(top_features, 2))
interactions3 = list(combinations(top_features, 3))
all_df_new = all_df.copy()
new_features = []
# compute product of all pairs of top features
for duo in interactions2:
    f1, f2 = duo
    v = all_df_new[f1] * all_df_new[f2]
    k = f"{f1}_{f2}_double_pos"
    all_df_new[k] = v
    new_features.append(k)
# compute combination features for all triplets of top features
for trio in interactions3:
    f1, f2, f3 = trio
    # compute a "triple neg" feature which is 1 if and only if
    # all 3 features are 0 aka if and only if all 3 genes are wildtype
    v = (all_df_new[f1] + all_df_new[f2] + all_df_new[f3]
         ).replace({0: 1, 1: 0, 2: 0, 3: 0})
    k = f"{f1}_{f2}_{f3}_triple_neg"
    all_df_new[k] = v
    new_features.append(k)
    # compute a "triple positive" features which corresponds
    # to the product of this feature triplet
    # aka it is 1 if all 3 genes are mutated
    v = (all_df_new[f1] * all_df_new[f2] * all_df_new[f3])
    k = f"{f1}_{f2}_{f3}_triple_pos"
    all_df_new[k] = v
    new_features.append(k)

# add multiple mutation features to the feature matrix
new_features.extend(multiple_mut.columns)
all_df_new = pd.merge(all_df_new, multiple_mut,
                      left_index=True,
                      right_index=True,
                      how="left")
all_df_new = all_df_new.fillna(0)

# compute chi2 (on binary labels) for all new features
# chi2 stat is computed exclusively on the trainig set to prevent leakage
X_train_new = all_df_new.loc[train_models, feature_col.tolist() + new_features]
X_test_new = all_df_new.loc[test_models, feature_col.tolist() + new_features]
chi2_stat, pval = [pd.Series(arr) for arr in chi2(X_train_new.values, y_train)]
chi2_df = pd.concat([chi2_stat, pval], axis=1)
chi2_df.index = X_train_new.columns
chi2_df.columns = ['chi2_stat', 'Pval']
chi2_df = chi2_df.sort_values('chi2_stat', ascending=False)

# filter new features by chi2 stat
features_tokeep = feature_col.tolist()
chi2_new_df = chi2_df.copy()
for gene in reversed(top_features.tolist()):  # inverse rank by chi2
    # pick the best (chi2) feature duo (aka double positive feature combo) involving gene
    gene_duos = chi2_new_df[(chi2_new_df.index.str.contains(gene)) & (
        chi2_new_df.index.str.contains('_double_'))]
    # check if there's any remaining duos involving gene
    if len(gene_duos) > 0:
        best_duo = gene_duos.index[0]
        # drop all other duos involving gene
        duos_todrop = gene_duos.index[1:].tolist()
    # pick best trio (aka triple positive or triple negative feature combo) involving gene
    gene_trios = chi2_new_df[(chi2_new_df.index.str.contains(gene)) & (
        chi2_new_df.index.str.contains('_triple_'))]
    # check if there's any remaining trios involving gene 
    if len(gene_trios) >0:
        best_trio = gene_trios.index[0]
        trios_todrop = gene_trios.index[1:].tolist()
        # dropa all other trios
        chi2_new_df = chi2_new_df.drop(duos_todrop + trios_todrop)

# save preprocessed features to file
features_tokeep = chi2_new_df.index.tolist()
all_df_new[features_tokeep].to_csv(snakemake.output.preproc_mut,
                                   sep='\t')
