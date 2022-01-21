#!/usr/bin/env python
# coding: utf-8

# # Introduction
# read and preprocess meth probe clusters

# ### Imports
# Import libraries and write settings here.
import pandas as pd
import numpy as np
from scipy import stats

# load pre-computed meth cluster labels
# TODO re compute on train set only to prevent info leaks
f = snakemake.input.meth_K5 
k5_clusters = pd.read_csv(f, sep="\t", header=0)
# convert index to CRC id short
k5_clusters.index = [c[:-3] for c in k5_clusters.index.tolist()]
# encode cluster labels as binary features
k5_clusters = pd.get_dummies(k5_clusters.cluster)

# load sample id conversion table, drug response data
target_col = snakemake.params.target_col
f = snakemake.input.response
drug_response_data = pd.read_csv(f, sep="\t")

# merge input, response df        
input_matrix = pd.merge(drug_response_data[[
                            target_col, "ircc_id_short", "ircc_id"]],
                            k5_clusters,
                            left_on="ircc_id_short",
                            right_index=True).\
                                    set_index('ircc_id').drop('ircc_id_short', axis=1)
features_col = np.array([c for c in input_matrix.columns if c != target_col])
# save processed features
f = snakemake.output.preproc_meth
input_matrix[features_col].to_csv(f,sep='\t')
