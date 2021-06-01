#!/usr/bin/env python
# coding: utf-8

# Data manipulation
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

# load sample id conversion table, drug response data
drug_response_data = pd.read_csv(snakemake.input.response, sep="\t")

# load driver annotation for PDx models
driver_data = pd.read_csv(snakemake.input.mut,
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

# 1-hot encode genes, vector sum on sample to
features_in = pd.get_dummies(features_pre.Gene)
features_in["ircc_id"] = features_pre.ircc_id
# account for multiple mut in same sample
features_in = features_in.groupby("ircc_id").sum()

target_col = snakemake.params.target_col
Mut = features_in
Y = drug_response_data
# encode target
Y_class_dict = {'PD': 0, 'SD': 1, 'OR': 1}
Y[target_col] = Y[target_col].replace(Y_class_dict)

feature_col = Mut.columns
all_df = pd.merge(Mut, Y[target_col], right_index=True,
                  left_index=True, how="right")
all_df = all_df.dropna(axis=0, how='all')
# fillna in features with median imputation
all_df[feature_col] = all_df[feature_col].astype(
    float).apply(lambda col: col.fillna(col.median()))
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]

# scale features
all_df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(all_df.values),
                             columns=all_df.columns,
                             index=all_df.index)

# train-test split
train_models = Y[Y.is_test == False].index.unique()
test_models = Y[Y.is_test == True].index.unique()
X_train = all_df_scaled.loc[train_models, feature_col].values
y_train = all_df_scaled.loc[train_models, target_col].values
X_test = all_df_scaled.loc[test_models, feature_col].values
y_test = all_df_scaled.loc[test_models, target_col].values


# perform univariate chi2 to establish a baseline of feature importance
# for engineering feature combos
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

# these features are 0 when there's 0 or 1 mut in gene, else
# they hold the count of unique mut per gene
multiple_mut = pre.replace({1: np.nan}).dropna(axis=1, how='all').fillna(0)
multiple_mut.columns = [c+"_MultiMut" for c in multiple_mut.columns]

# count tot unique mut burden per sample
multiple_mut["unique_mut_burden"] = pre.apply(pd.Series.sum, axis=1)
multiple_mut.shape

# use only top gene features (by univariate chi2 stat)
interactions2 = list(combinations(top_features, 2))
interactions3 = list(combinations(top_features, 3))

all_df_new = all_df_scaled.copy()

new_features = []
# build combos of 2 features
for duo in interactions2:
    f1, f2 = duo
    v = all_df_new[f1] * all_df_new[f2]
    k = f"{f1}_{f2}_double_pos"
    all_df_new[k] = v
    new_features.append(k)
# build combos of 3 features
for trio in interactions3:
    f1, f2, f3 = trio
    v = (all_df_new[f1] + all_df_new[f2] + all_df_new[f3]
         ).replace({0: 1, 1: 0, 2: 0, 3: 0})
    k = f"{f1}_{f2}_{f3}_triple_neg"
    all_df_new[k] = v
    new_features.append(k)
    v = (all_df_new[f1] * all_df_new[f2] * all_df_new[f3])
    k = f"{f1}_{f2}_{f3}_triple_pos"
    all_df_new[k] = v
    new_features.append(k)

# add MultiMut features aka multiple muts on same gene
new_features.extend(multiple_mut.columns)
all_df_new = pd.merge(all_df_new, multiple_mut,
                      left_index=True,
                      right_index=True,
                      how="left")
all_df_new = all_df_new.fillna(0)
# standardise features [0,1]
all_df_new_scaled = pd.DataFrame(MinMaxScaler().fit_transform(all_df_new.values),
                                 columns=all_df_new.columns,
                                 index=all_df_new.index)

X_train_new = all_df_new.loc[train_models, feature_col.tolist() + new_features]
X_test_new = all_df_new.loc[test_models, feature_col.tolist() + new_features]

chi2_stat, pval = [pd.Series(arr) for arr in chi2(X_train_new.values, y_train)]
chi2_df = pd.concat([chi2_stat, pval], axis=1)
chi2_df.index = X_train_new.columns
chi2_df.columns = ['chi2_stat', 'Pval']
chi2_df = chi2_df.sort_values('chi2_stat', ascending=False)
chi2_df.head(30)

features_tokeep = feature_col.tolist()
chi2_new_df = chi2_df.copy()
for gene in reversed(top_features.tolist()):  # inverse rank by chi2
    # pick the best (chi2) feature duo involving gene
    gene_duos = chi2_new_df[(chi2_new_df.index.str.contains(gene)) & (
        chi2_new_df.index.str.contains('_double_'))]
    best_duo = gene_duos.index[0]
    duos_todrop = gene_duos.index[1:].tolist()  # drop the others
    # pick best trio
    gene_trios = chi2_new_df[(chi2_new_df.index.str.contains(gene)) & (
        chi2_new_df.index.str.contains('_triple_'))]
    best_trio = gene_trios.index[0]
    trios_todrop = gene_trios.index[1:].tolist()
    # drop unselected engineered features
    chi2_new_df = chi2_new_df.drop(duos_todrop + trios_todrop)

features_tokeep = chi2_new_df.index.tolist()
all_df_new_scaled[features_tokeep].to_csv(snakemake.output.preproc_mut,
                                          sep='\t')
