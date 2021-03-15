#!/usr/bin/env python
# coding: utf-8
# Data manipulation
import pandas as pd
import numpy as np

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# load train set for each omic
Meth_train = pd.read_csv(snakemake.input.meth_Xtrain,
                         sep="\t", header=0, index_col=0)
Expr_train = pd.read_csv(snakemake.input.expr_Xtrain,
                         sep="\t", header=0, index_col=0)
Expr_train.columns = [c + "_expr" for c in Expr_train.columns]
CNV_train = pd.read_csv(snakemake.input.cnv_Xtrain,
                        sep="\t", header=0, index_col=0)
CNV_train.columns = [c + "_cnv" for c in CNV_train.columns]
MutVAF_train = pd.read_csv(snakemake.input.mutVAF_Xtrain,
                           sep="\t", header=0, index_col=0)
MutVAF_train.columns = [c + "_mutVAF" for c in MutVAF_train.columns]
Mut_train = pd.read_csv(snakemake.input.mut_Xtrain,
                        sep="\t", header=0, index_col=0)
Mut_train.columns = [c + "_mut" for c in Mut_train.columns]
# load response data for the training set
target_col = snakemake.params.target_col
drug_response_data = pd.read_csv(snakemake.input.response,
                                 sep="\t")
Y_train = drug_response_data[drug_response_data.is_test == False][
    [target_col,
     "ircc_id"]].set_index("ircc_id")

# merge all train sets
df1 = pd.merge(Mut_train, CNV_train, right_index=True,
               left_index=True, how="outer")
df2 = pd.merge(df1, MutVAF_train, right_index=True,
               left_index=True, how="outer")
df3 = pd.merge(Meth_train, Expr_train, right_index=True,
               left_index=True, how="outer")
all_df = pd.merge(df3, df2, right_index=True, left_index=True, how="outer")
# merge with the train target set
all_df = pd.merge(all_df, Y_train,
                  right_index=True, left_index=True, how="right")


feature_col = [c for c in all_df.columns if c != target_col]
# fillna in features with median imputation
all_df[feature_col] = all_df[feature_col].apply(
    lambda col: col.fillna(col.median()))
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]

# save individual omic files, log shape
arr = []
Meth_train_clean = all_df[Meth_train.columns]
Meth_train_clean.to_csv(snakemake.output.meth_Xtrain, sep="\t")
arr.append(["Meth_train", len(Meth_train.columns), len(Meth_train_clean)])

Expr_train_clean = all_df[Expr_train.columns]
Expr_train_clean.to_csv(snakemake.output.expr_Xtrain, sep="\t")
arr.append(["Expr_train", len(Expr_train.columns), len(Expr_train_clean)])

CNV_train_clean = all_df[CNV_train.columns]
CNV_train_clean.to_csv(snakemake.output.cnv_Xtrain, sep="\t")
arr.append(["CNV_train", len(CNV_train.columns), len(CNV_train_clean)])

Mut_train_clean = all_df[Mut_train.columns]
Mut_train_clean.to_csv(snakemake.output.mut_Xtrain, sep="\t")
arr.append(["Mut_train", len(Mut_train.columns), len(Mut_train_clean)])

MutVAF_train_clean = all_df[MutVAF_train.columns]
MutVAF_train_clean.to_csv(snakemake.output.mutVAF_Xtrain, sep="\t")
arr.append(["MutVAF_train", len(MutVAF_train.columns), len(MutVAF_train_clean)])

Y_train_clean = all_df[target_col]
Y_train_clean.to_csv(snakemake.output.Ytrain, sep="\t")
arr.append(["Y_train", 1, len(Y_train_clean)])

log_df = pd.DataFrame(arr, columns=["dataset", "n_features", "n_instances"])
log_df.to_csv(snakemake.log[0], sep="\t")
