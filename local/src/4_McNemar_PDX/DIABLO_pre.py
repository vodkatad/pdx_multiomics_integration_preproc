#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Prepare mixOmics PLS-DA / DIABLO input
# starting from 'raw' non-engineered multi-omic features

# ### Imports
# Import libraries and write settings here.
# Data manipulation
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import warnings
# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30
# scalers
# models
# feature selection

# processing
# benchmark
# Hyperparameter tuning
# model persistence


# Analysis/Modeling
# load all 'omics preprocessed datasets
# raw methylation probes data
f = snakemake.input.meth
Meth = pd.read_csv(f, sep="\t", header=0, index_col=0)
Meth = Meth[Meth.columns.drop(list(Meth.filter(regex='(Cetuximab)|(_id)')))]
# raw expression data (variance-stabilised RNAseq)
f = snakemake.input.expr
Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr = Expr[Expr.columns.drop(list(Expr.filter(regex='(Cetuximab)|(_id)')))]
Expr.columns = [c + "_expr" for c in Expr.columns]
# binary CNV features includes
# loss, gain highGain (> than 1.5 copies gained) events
f = snakemake.input.cnv
CNV = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV = CNV[CNV.columns.drop(list(CNV.filter(regex='(Cetuximab)|(_id)')))]
CNV.columns = [c + "_cnv" for c in CNV.columns]
# binary cancer driver mutation events (mutations per gene aggregated)
f = snakemake.input.mut
Mut = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut = Mut[Mut.columns.drop(list(Mut.filter(regex='(Cetuximab)|(_id)')))]
Mut.columns = [c + "_mut" for c in Mut.columns]
# clinical data on origin patient
# extensive preproc, egineering done here but no clustering/cross
f = snakemake.input.clin
Clin = pd.read_csv(f, sep="\t", header=0, index_col=0)
Clin = Clin[Clin.columns.drop(list(Clin.filter(regex='(Cetuximab)|(_id)')))]
Clin.columns = [c + "_clin" for c in Clin.columns]
# load drug response data
f = snakemake.input.response
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode target var (binary responder/non-responder)
target_col = snakemake.params.target_col
Y_class_dict = {'PD': 0, 'OR+SD': 1}
Y[target_col] = Y[target_col].replace(Y_class_dict)

# merge all feature blocks + response together
df1 = pd.merge(Mut, CNV, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth, Expr, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
all_df = pd.merge(all_df, Clin, right_index=True, left_index=True, how="outer")
# drop all id cols except index
all_df = all_df[all_df.columns.drop(list(all_df.filter(regex='_id')))]
feature_col = all_df.columns.tolist()
# force to numeric
all_df = all_df.select_dtypes([np.number])
# add target
all_df = pd.merge(all_df, Y[target_col],
                  right_index=True,
                  left_index=True,
                  how="right")
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]

# train-test split
train_models = Y[Y.is_test == False].index.unique()
test_models = Y[Y.is_test == True].index.unique()
X_train = all_df.loc[train_models, feature_col]
y_train = all_df.loc[train_models, target_col]
X_test = all_df.loc[test_models, feature_col]
y_test = all_df.loc[test_models, target_col]
# scale features separately, fill sparse features w/t median imputation
scaler = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train.values),
                       columns=X_train.columns, index=X_train.index)
X_train[feature_col] = X_train[feature_col].\
    astype(float).apply(lambda col: col.fillna(col.median()))
X_test = pd.DataFrame(scaler.transform(X_test.values),
                      columns=X_test.columns, index=X_test.index)
X_test[feature_col] = X_test[feature_col].\
    astype(float).apply(lambda col: col.fillna(col.median()))
# log train, test shape, dataset balance
logfile = snakemake.log.lasso_select
with open(logfile, "w") as log:
    log.write(
        f"There are {X_train.shape[0]} instances in the trainig set.\n")
    log.write(f"There are {X_test.shape[0]} instances in the test set.\n")
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    log.write(f"There are {train_counts.loc[0]} 'PD' instances and "
              f"{train_counts.loc[1]} 'SD+OR' instances in the training set.\n")
    log.write(f"There are {test_counts.loc[0]} 'PD' instances and "
              f"{test_counts.loc[1]} 'SD+OR' instances in the test set.\n")
# log n of features for each block
with open(logfile, "a") as log:
    log.write(f"There are {X_train.shape[1]} total features." + '\n')
    log.write(f"There are {Meth.shape[1]} methylation features." + '\n')
    log.write(f"There are {Expr.shape[1]} expression features." + '\n')
    log.write(f"There are {Mut.shape[1]} mutation features." + '\n')
    log.write(f"There are {CNV.shape[1]} copy number features." + '\n')
    log.write(f"There are {Clin.shape[1]} clinical features." + '\n')

# write test sets to file
y_train.to_csv(snakemake.output.Y_train, sep='\t')
y_test.to_csv(snakemake.output.Y_test, sep='\t')

# re-split X into omics blocks for mixOmics
X_train[Mut.columns].to_csv(
    snakemake.output.mut_train, sep='\t')
X_test[Mut.columns].to_csv(
    snakemake.output.mut_test, sep='\t')
X_train[CNV.columns].to_csv(
    snakemake.output.cnv_train, sep='\t')
X_test[CNV.columns].to_csv(
    snakemake.output.cnv_test, sep='\t')
X_train[Clin.columns].to_csv(
    snakemake.output.clin_train, sep='\t')
X_test[Clin.columns].to_csv(
    snakemake.output.clin_test, sep='\t')

# run Lasso (L1 -penalised LogReg) feature selection on methylatioin and expression data
L1LR = LogisticRegression(
    C=.9,
    penalty='l1',
    random_state=13,
    solver='saga',
    max_iter=500)
L1Selector = SelectFromModel(estimator=L1LR)

expr_train = X_train[Expr.columns]
expr_test = X_test[Expr.columns]
features = expr_train.columns.tolist()
selector = L1Selector.fit(expr_train, y_train)
trsh = selector.threshold_
support = selector.get_support()
selected_features = pd.Series(features)[support]
coef = selector.estimator_.coef_[0]
arr = []
for f, s, c in zip(features, support, coef):
    arr.append([f, s, c])
pd.DataFrame(arr, columns=['features', 'support', 'coef']).to_csv(
    snakemake.log.expr_coef, sep='\t')
with open(snakemake.log.lasso_select, 'a') as logfile:
    logfile.write(f'L1-penalised LR selected {len(selected_features)}'
                  f' expression features out of {len(features)}'
                  f' using a {trsh: .3f} threshold.\n')
expr_train[selected_features].to_csv(
    snakemake.output.expr_train, sep='\t')
expr_test[selected_features].to_csv(
    snakemake.output.expr_test, sep='\t')

meth_train = X_train[Meth.columns]
meth_test = X_test[Meth.columns]
features = meth_train.columns.tolist()
selector = L1Selector.fit(meth_train, y_train)
trsh = selector.threshold_
support = selector.get_support()
selected_features = pd.Series(features)[support]
coef = selector.estimator_.coef_[0]
arr = []
for f, s, c in zip(features, support, coef):
    arr.append([f, s, c])
pd.DataFrame(arr, columns=['features', 'support', 'coef']).to_csv(
    snakemake.log.meth_coef, sep='\t')
with open(snakemake.log.lasso_select, 'a') as logfile:
    logfile.write(f'L1-penalised LR selected {len(selected_features)}'
                  f' methylation features out of {len(features)}'
                  f' using a {trsh: .3f} threshold.\n')
meth_train[selected_features].to_csv(
    snakemake.output.meth_train, sep='\t')
meth_test[selected_features].to_csv(
    snakemake.output.meth_test, sep='\t')
