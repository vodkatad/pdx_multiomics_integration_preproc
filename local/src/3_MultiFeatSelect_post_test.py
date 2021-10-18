#!/usr/bin/env python
# coding: utf-8
# Data manipulation
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None


# load train set for each omic
Meth_test = pd.read_csv(snakemake.input.meth_Xtest,
                        sep="\t", header=0, index_col=0)
Expr_test = pd.read_csv(snakemake.input.expr_Xtest,
                        sep="\t", header=0, index_col=0)
Expr_test.columns = [c + "_expr" for c in Expr_test.columns]
CNV_test = pd.read_csv(snakemake.input.cnv_Xtest,
                       sep="\t", header=0, index_col=0)
CNV_test.columns = [c + "_cnv" for c in CNV_test.columns]
MutVAF_test = pd.read_csv(snakemake.input.mutVAF_Xtest,
                          sep="\t", header=0, index_col=0)
MutVAF_test.columns = [c + "_mutVAF" for c in MutVAF_test.columns]
Mut_test = pd.read_csv(snakemake.input.mut_Xtest,
                       sep="\t", header=0, index_col=0)
Mut_test.columns = [c + "_mut" for c in Mut_test.columns]
# load response data for the training set
target_col = snakemake.params.target_col
drug_response_data = pd.read_csv(snakemake.input.response,
                                 sep="\t")
Y_test = drug_response_data[drug_response_data.is_test == True][
    [target_col,
     "ircc_id"]].set_index("ircc_id")

# merge all test sets
df1 = pd.merge(Mut_test, CNV_test, right_index=True,
               left_index=True, how="outer")
df2 = pd.merge(df1, MutVAF_test, right_index=True,
               left_index=True, how="outer")
df3 = pd.merge(Meth_test, Expr_test, right_index=True,
               left_index=True, how="outer")
all_df = pd.merge(df3, df2, right_index=True, left_index=True, how="outer")
# merge with the test target set
all_df = pd.merge(all_df, Y_test,
                  right_index=True, left_index=True, how="right")


feature_col = [c for c in all_df.columns if c != target_col]
# fillna in features with median imputation
all_df[feature_col] = all_df[feature_col].apply(
    lambda col: col.fillna(col.median()))
# re-stanardise all features together (including binary features)
all_df[feature_col] = pd.DataFrame(StandardScaler().
                                   fit_transform(all_df[feature_col].values),
                                   columns=feature_col,
                                   index=all_df.index)
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]

# save individual omic files, log shape
arr = []
Meth_test_clean = all_df[Meth_test.columns]
Meth_test_clean.to_csv(snakemake.output.meth_Xtest, sep="\t")
arr.append(["Meth_test", len(Meth_test.columns), len(Meth_test_clean)])

Expr_test_clean = all_df[Expr_test.columns]
Expr_test_clean.to_csv(snakemake.output.expr_Xtest, sep="\t")
arr.append(["Expr_test", len(Expr_test.columns), len(Expr_test_clean)])

CNV_test_clean = all_df[CNV_test.columns]
CNV_test_clean.to_csv(snakemake.output.cnv_Xtest, sep="\t")
arr.append(["CNV_test", len(CNV_test.columns), len(CNV_test_clean)])

Mut_test_clean = all_df[Mut_test.columns]
Mut_test_clean.to_csv(snakemake.output.mut_Xtest, sep="\t")
arr.append(["Mut_test", len(Mut_test.columns), len(Mut_test_clean)])

MutVAF_test_clean = all_df[MutVAF_test.columns]
MutVAF_test_clean.to_csv(snakemake.output.mutVAF_Xtest, sep="\t")
arr.append(["MutVAF_test", len(MutVAF_test.columns), len(MutVAF_test_clean)])

Y_test_clean = all_df[target_col]
Y_test_clean.to_csv(snakemake.output.Ytest, sep="\t")
arr.append(["Y_test", 1, len(Y_test_clean)])

log_df = pd.DataFrame(arr, columns=["dataset", "n_features", "n_instances"])
log_df.to_csv(snakemake.log[0], sep="\t")
