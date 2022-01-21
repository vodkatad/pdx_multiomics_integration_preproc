#!/usr/bin/env python
# coding: utf-8

# ### Imports

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# load sample id conversion table, genealogy data
id_data = pd.read_csv(snakemake.input.ids, sep="\t", names=[
                      "sanger_id", "ircc_id", "sample_level"])
# eval if sample is liver metastasis in PDx
id_data["is_LMX"] = id_data.ircc_id.str.contains("LMX")
# short ids consist of the first 7 chars of full ids
id_data["ircc_id_short"] = id_data.ircc_id.apply(lambda x: x[0:7])
id_data["passage"] = id_data.ircc_id.apply(lambda x: int(x[12:14]))
id_data["lineage"] = id_data.ircc_id.apply(lambda x: x[10:12])
# filter out non-LMX models, keep first level models only
id_data = id_data[(id_data.is_LMX == True) &
                  (id_data.sample_level == "firstlevel")]

# load drug response data for PDx models
target_col = snakemake.params.target_col
target_col_join = target_col.replace(" ", "_")
drug_response_data = pd.read_csv(snakemake.input.drug, "\t", header=0).\
    rename(columns={'CRC CODE':
                    'ircc_id_short',
                    target_col:
                    target_col_join})
# merge response data w/t gnenealogy data
drug_response_data = pd.merge(id_data,
                              drug_response_data[[
                                  target_col_join, "ircc_id_short"]],
                              on="ircc_id_short")
# drop models w/t missing target variable value
drug_response_data = drug_response_data[~drug_response_data.Cetuximab_Standard_3wks.isna(
)]

# encode the continuous target variable (tumour growth after Cetuximab treatment)
# as a binary var with two states: 
# Progressive Disease and
# Stable Disease-Objective Response
bins = [np.NINF] + snakemake.params.class_bins + [np.Infinity]
target_col_cat = target_col_join+"_cat"
drug_response_data[target_col_cat] = pd.cut(
    drug_response_data[target_col_join],
    bins=bins,
    labels=snakemake.params.class_labels)

# generate train/test split shuffle (stratified) replicates
# returns stratified randomized folds preserving 
# the percentage of samples for each class.
test_size = float(snakemake.params.testSize)
n_splits = int(snakemake.params.n_splits) 
labels = drug_response_data[target_col_cat]
all_models = drug_response_data.ircc_id.unique()
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=.3, random_state=13)
for duo, outfile in zip(sss.split(all_models, labels), snakemake.output):
    out_df = drug_response_data.copy() 
    train_index, test_index = duo
    train_models = all_models[train_index]
    test_models = all_models[test_index] 
    out_df["is_test"] = out_df["ircc_id"].isin(test_models)
    out_df.to_csv(outfile, sep="\t", index=None)
