#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Imports
# Import libraries and write settings here.

# Data manipulation
import pandas as pd
import numpy as np
from itertools import combinations

# feature scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# feature selection
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2


# load PDX sample id conversion table, drug response data
f = snakemake.input.PDX_response
drug_response_data = pd.read_csv(f, sep="\t")
# load driver annotation for PDx models
f = snakemake.input.PDX_mut
driver_data = pd.read_csv(f, "\t", header=0).rename(columns={'Sample':
                                                             'sanger_id'})
# join response, driver annot data
drug_mut_df = pd.merge(drug_response_data, driver_data,
                       on="sanger_id", how="left")
# transform to have a gene x sample binary mutation matrix
# including all driver genes
features_pre = drug_mut_df[drug_mut_df["Final_Driver_Annotation"] == True][
    ["ircc_id",
     "Gene"]
].drop_duplicates()

# 1-hot encode genes, vector sum on sample to
PDX_features_in = pd.get_dummies(features_pre.Gene)
PDX_features_in["ircc_id"] = features_pre.ircc_id
# account for multiple mut in same sample
PDX_features_in = PDX_features_in.groupby("ircc_id").sum()
PDX_features = PDX_features_in.columns.tolist()
target_col = snakemake.params.PDX_target_col
Mut = PDX_features_in

# encode PDX model target variable (Cetuximab response 3w)
# into a binary responder/non-responder var
Y = drug_response_data.set_index('ircc_id')
Y_dict = {"OR+SD": 1, "PD": 0}
Y[target_col] = Y[target_col].replace(Y_dict)
# feature set labels
feature_col = Mut.columns
# merge target, features along sample names
PDX_all_df = pd.merge(Mut, Y[target_col],
                      right_index=True, left_index=True, how="right")
PDX_all_df = PDX_all_df.dropna(axis=0, how='all')
# fillna in features with median imputation
PDX_all_df[feature_col] = PDX_all_df[feature_col].astype(float).\
    apply(lambda col: col.fillna(col.median()))
# drop duplicated instances (ircc_id) from index
PDX_all_df = PDX_all_df[~PDX_all_df.index.duplicated(keep='first')]
# train-test split
# here concatenate PDX train and test X ioon order to transform
# both at the same time with the new CMP mutation features engineered from the
# CMP train set
PDX_all_models = Y.index.unique()
PDX_X_test = PDX_all_df.loc[PDX_all_models, feature_col]
PDX_y_test = PDX_all_df.loc[PDX_all_models, target_col]


# load CMP mutation data
f = snakemake.input.CMP_mut
CMP_mut_data = pd.read_csv(f, header=0)
# load GDSC drug response data
f = snakemake.input.CMP_response
drug_response_data = pd.read_csv(f, sep="\t", index_col=None)
all_models = drug_response_data.SANGER_MODEL_ID.values
# slice pre-selected CRC models
CMP_mut_data = CMP_mut_data[CMP_mut_data.model_id.isin(all_models)]
# get gene overlap between the CMP mut data and PDX
CRC_cetux_mut_gene = CMP_mut_data.gene_symbol.unique()
PDX_CRC_cetux_mut_gene = set(PDX_features).intersection(CRC_cetux_mut_gene)
# slice shared genes (gene overlap), keep only protein mut
CMP_mut_data = CMP_mut_data[(CMP_mut_data.gene_symbol.isin(PDX_CRC_cetux_mut_gene)) &
                            (CMP_mut_data.protein_mutation != '-')]
# transform df to have a gene x sample binary mutation matrix
# including all driver genes
features_pre = CMP_mut_data[['gene_symbol', 'model_id']
                            ].drop_duplicates()
# 1-hot encode genes, vector sum on sample
CMPGDSC_features_in = pd.get_dummies(features_pre.gene_symbol)
CMPGDSC_features_in["model_id"] = features_pre.model_id
# account for multiple mut in same sample
CMPGDSC_features_in = CMPGDSC_features_in.groupby("model_id").sum()
target_col = snakemake.params.CMP_target_col
Mut = CMPGDSC_features_in
Y = drug_response_data[['SANGER_MODEL_ID',
                        target_col]].set_index('SANGER_MODEL_ID')
Y[target_col] = Y[target_col].replace(Y_dict)
feature_col = Mut.columns
CMPGDSC_all_df = pd.merge(Mut, Y[target_col],
                          right_index=True, left_index=True, how="right")
CMPGDSC_all_df = CMPGDSC_all_df.dropna(axis=0, how='all')
# drop duplicated instances (SANGER_ID) from index
CMPGDSC_all_df = CMPGDSC_all_df[~CMPGDSC_all_df.index.duplicated(keep='first')]
# use the full CMP-GDSC dataset as training set
CMPGDSC_X_train = CMPGDSC_all_df.loc[:, feature_col]
CMPGDSC_y_train = CMPGDSC_all_df.loc[:, target_col]

# ### Combine CMP training set and PDX test set
# Add any missing features as [0,..., 0], scale independently

# combine train, test datasets
train_models = CMPGDSC_X_train.index.tolist()
test_models = PDX_X_test.index.tolist()
all_df = pd.concat([CMPGDSC_X_train, PDX_X_test])
Y = pd.concat([CMPGDSC_y_train, PDX_y_test])
# fill any missing feature with 0s
feature_col = all_df.columns.tolist()
all_df = all_df.fillna(0)
# fit scaler, scale independently
X_train = all_df.loc[train_models, feature_col]
X_test = all_df.loc[test_models, feature_col]
y_train = Y.loc[train_models]
y_test = Y.loc[test_models]
X_train = pd.DataFrame(MinMaxScaler().fit_transform(X_train.values),
                       columns=X_train.columns,
                       index=X_train.index)
X_test = pd.DataFrame(MinMaxScaler().fit_transform(X_test.values),
                      columns=X_test.columns,
                      index=X_test.index)
all_df_scaled = pd.concat([X_train, X_test])

# ### Feature Engineering

# univariate chi2 to establish a baseline for feature combos
print('selecting top features for crosses')
chi2_stat, pval = [pd.Series(arr) for arr in chi2(X_train, y_train)]
chi2_df = pd.concat([chi2_stat, pval], axis=1)
chi2_df.index = feature_col
chi2_df.columns = ['chi2_stat', 'Pval']
chi2_df = chi2_df.sort_values('chi2_stat', ascending=False)
pctl_tr = chi2_df.chi2_stat.describe().loc['75%']  # set a chi2 pctl threshold
# get top gene features sorted by chi2 stat
top_features = chi2_df[(chi2_df.chi2_stat > pctl_tr)].index

# build new features by counting the unique number of protein mut per gene per sample
pre_PDX = drug_mut_df[drug_mut_df.Gene.isin(feature_col)][
    ["ircc_id",
     "Gene",
     "Protein"]
].drop_duplicates().groupby(["ircc_id", "Gene"]).Protein.nunique().\
    unstack().loc[test_models, :]
pre_CMPGDSC = CMP_mut_data[CMP_mut_data.gene_symbol.isin(feature_col)][[
    'model_id',
    'gene_symbol',
    'protein_mutation'
]].drop_duplicates().groupby(["model_id", "gene_symbol"]).protein_mutation.nunique().\
    unstack().loc[train_models, :]
# rename indeces
pre_PDX.index = pre_PDX.index.tolist()
pre_PDX.columns = pre_PDX.columns.tolist()
pre_CMPGDSC.index = pre_CMPGDSC.index.tolist()
pre_CMPGDSC.columns = pre_CMPGDSC.columns.tolist()
pre = pd.concat([pre_CMPGDSC, pre_PDX])

# these MultiMut features are 0 when there's 0 or 1 mut in gene, else
# they hold the count of unique mut per gene
multiple_mut = pre.replace({1: np.nan}).dropna(axis=1, how='all').fillna(0)
multiple_mut.columns = [c+"_MultiMut" for c in multiple_mut.columns]
# count tot unique mut burden per sample
multiple_mut["unique_mut_burden"] = pre.apply(pd.Series.sum, axis=1)

# compute 2x and 3x feature crosses
# use only top gene features
print('computing feture crosses')
interactions2 = list(combinations(top_features, 2))
interactions3 = list(combinations(top_features, 3))
all_df_new = all_df_scaled.copy()
new_features = []
for duo in interactions2:
    f1, f2 = duo
    v = all_df_new[f1] * all_df_new[f2]
    k = f"{f1}_{f2}_double_pos"
    all_df_new[k] = v
    new_features.append(k)
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

# add MultiMut features
# aka are there multiple muts on this gene in this sample?
new_features.extend(multiple_mut.columns)
all_df_new = pd.merge(all_df_new, multiple_mut,
                      left_index=True,
                      right_index=True,
                      how="left")
all_df_new = all_df_new.fillna(0)
# standardise all new features
all_df_new_scaled = pd.DataFrame(MinMaxScaler().fit_transform(all_df_new.values),
                                 columns=all_df_new.columns,
                                 index=all_df_new.index)

# build a new training, test dataset including all new features
X_train_new = all_df_new.loc[train_models, feature_col + new_features]
X_test_new = all_df_new.loc[test_models, feature_col + new_features]
chi2_stat, pval = [pd.Series(arr) for arr in chi2(X_train_new.values, y_train)]
chi2_df = pd.concat([chi2_stat, pval], axis=1)
chi2_df.index = X_train_new.columns
chi2_df.columns = ['chi2_stat', 'Pval']
chi2_df = chi2_df.sort_values('chi2_stat', ascending=False)

# since there are multiple crosses containing each gene name,
# pick the best 3x and 2x cross involving said gene
features_tokeep = feature_col
chi2_new_df = chi2_df.copy()
for gene in reversed(top_features.tolist()):  # inverse rank by chi2
    # pick the best (chi2) feature duo involving gene
    gene_duos = chi2_new_df[(chi2_new_df.index.str.contains(gene)) &
                            (chi2_new_df.index.str.contains('_double_'))]
    try:
        best_duo = gene_duos.index[0]
    except IndexError:
        continue
    duos_todrop = gene_duos.index[1:].tolist()  # drop the others
    # pick best trio involving gene
    gene_trios = chi2_new_df[(chi2_new_df.index.str.contains(gene)) &
                             (chi2_new_df.index.str.contains('_triple_'))]
    try:
        best_trio = gene_trios.index[0]
    except IndexError:
        continue
    trios_todrop = gene_trios.index[1:].tolist()
    # drop unselected features
    chi2_new_df = chi2_new_df.drop(duos_todrop + trios_todrop)
features_tokeep = chi2_new_df.index.tolist()
# save selected features to file
all_df_new_scaled[features_tokeep].to_csv(snakemake.output.preproc_mut,
                                          sep='\t')
