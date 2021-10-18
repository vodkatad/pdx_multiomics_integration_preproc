#!/usr/bin/env python
# coding: utf-8

# ### Imports

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
# drop moels w/t missing target variable value
drug_response_data = drug_response_data[~drug_response_data.Cetuximab_Standard_3wks.isna(
)]

# encode target variable
# Progressive Disease, Stable Disease-Objective Response
bins = [np.NINF] + snakemake.params.class_bins + [np.Infinity]
target_col_cat = target_col_join+"_cat"
drug_response_data[target_col_cat] = pd.cut(
    drug_response_data[target_col_join],
    bins=bins,
    labels=snakemake.params.class_labels)

# train/test split (stratified)
test_size = float(snakemake.params.testSize)
labels = drug_response_data[target_col_cat].values
all_models = drug_response_data.ircc_id.unique()
train_models, test_models = train_test_split(
    all_models, test_size=test_size, stratify=labels)
drug_response_data["is_test"] = drug_response_data["ircc_id"].isin(test_models)
drug_response_data.to_csv(snakemake.output.response_tab, sep="\t", index=None)
