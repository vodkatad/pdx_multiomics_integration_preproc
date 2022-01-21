#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Train, optimise stacked predictor of Cetuximab sensitivity

# ### Imports
# Import libraries and write settings here.
# Data manipulation
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# load PDX test set
f = snakemake.input.PDX_X_test
PDX_X_test = pd.read_csv(f, sep='\t',
                         index_col=0, header=0)
PDX_features = PDX_X_test.columns.tolist()

# load all Charles River 'omics preprocessed datasets
# K5 clusters encoded meth probes
f = snakemake.input.meth
Meth = pd.read_csv(f, sep="\t", header=0).set_index("short_CR_id")
# encoded expr data w/t progeny pathway scores + msdb hallmarks ssGSEA scores
# processed through a colinearity + chi2 filter (drop the worst of each colinear pair of features)
f = snakemake.input.expr
Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr.columns = [c + "_expr" for c in Expr.columns]
# feature agglomeration CNV, input includes highGain features (> than 1 copy gained)
f = snakemake.input.cnv
CNV = pd.read_csv(f, sep="\t", header=0).set_index("short_CR_id")
CNV.columns = [c + "_cnv" for c in CNV.columns]
# custom mut feature cross w/t top 20 features by chi2
f = snakemake.input.mut
Mut = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut.columns = [c + "_mut" for c in Mut.columns]
# add clinical data (custom encoding, filtering)
f = snakemake.input.clin
Clin = pd.read_csv(f, sep="\t", header=0).set_index("short_CR_id")
Clin.columns = [c + "_clin" for c in Clin.columns]
# load drug response data
f = snakemake.input.response
Y = pd.read_csv(f, sep="\t").set_index("short_CR_id")
# encode target var (binary responder/non-responder)
target_col = snakemake.params.target_col
Y_class_dict = {'PD': 0, 'OR': 1, 'SD': 1}
Y[target_col] = Y[target_col].replace(Y_class_dict)
test_models = Y.index.tolist()

# merge all feature blocks + response together
df1 = pd.merge(Mut, CNV, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth, Expr, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
all_df = pd.merge(all_df, Clin, right_index=True, left_index=True, how="outer")
feature_col = all_df.columns.tolist()
# add target col
all_df = pd.merge(all_df, Y[target_col], right_index=True,
                  left_index=True, how="right")
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]
# fill sparse features with median imputation
all_df[feature_col] = all_df[feature_col].\
    astype(float).apply(lambda col: col.fillna(col.median()))
# use only CR PDX models that have Cetuximab response data
X_test = all_df.loc[test_models, PDX_features]
y_test = all_df.loc[test_models, target_col]
# scale features separately
scaler = MinMaxScaler().fit(X_test)
X_test = pd.DataFrame(scaler.transform(X_test.values),
                      columns=X_test.columns, index=X_test.index)
# log train, test shape, dataset balance
logfile = snakemake.log[0]
with open(logfile, "w") as log:
    log.write(f"There are {X_test.shape[0]} instances in the test set." + '\n')
    test_counts = y_test.value_counts()
    log.write(f"There are {test_counts.loc[0]} 'PD' instances and\
         {test_counts.loc[1]} 'SD+OR' instances in the test set." + '\n')
# get indeces for feature subsets, one per OMIC
Meth_indeces = list(range(0, Meth.shape[1]))
pos = len(Meth_indeces)
Expr_indeces = list(range(Meth_indeces[-1]+1, pos + Expr.shape[1]))
pos += len(Expr_indeces)
Mut_indeces = list(range(Expr_indeces[-1]+1, pos + Mut.shape[1]))
pos += len(Mut_indeces)
CNV_indeces = list(range(Mut_indeces[-1]+1, pos + CNV.shape[1]))
pos += len(CNV_indeces)
Clin_indeces = list(range(CNV_indeces[-1]+1, pos + Clin.shape[1]))
# log n of features for each block
with open(logfile, "a") as log:
    log.write(f"There are {X_test.shape[1]} total features." + '\n')
    log.write(f"There are {Meth.shape[1]} methylation features." + '\n')
    log.write(f"There are {Expr.shape[1]} expression features." + '\n')
    log.write(f"There are {Mut.shape[1]} mutation features." + '\n')
    log.write(f"There are {CNV.shape[1]} copy number features." + '\n')
    log.write(f"There are {Clin.shape[1]} clinical features." + '\n')

# write test set to file
X_test.to_csv(snakemake.output.X_test, sep='\t')
y_test.to_csv(snakemake.output.Y_test, sep='\t')
