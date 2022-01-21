#!/usr/bin/env python
# coding: utf-8

# ### Imports
# Data manipulation
import pandas as pd
import numpy as np
from scipy import stats
# scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# feature selection
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2


logfile = snakemake.log[0]

# ### Load PDX PROGENY, Hallmarks pre-filtered scores
f = snakemake.input.PDX_preproc_expr
PDX_combined = pd.read_csv(f, sep="\t").set_index('ircc_id')
PDX_features = PDX_combined.columns.tolist() 

# ### Load and combine Charles River PROGENy and MSdB Hallmarks features
# These have been computed over the full CR set
# load GDSC Cetuximab response data
f = snakemake.input.CR_response
drug_response_data = pd.read_csv(f,
                                 sep="\t",
                                 index_col=None)
all_models = drug_response_data.short_CR_id.values
CR_target_col = snakemake.params.CR_target_col
y_test = drug_response_data[['short_CR_id',
                             CR_target_col]].\
    set_index('short_CR_id')

# parse CR PROGENy scores for all instances
progeny_all = pd.read_csv(snakemake.input.CR_progeny,
                          header=0, index_col=0)
# add column prefix
progeny_all.columns = ['PROGENy_' + c for c in progeny_all.columns]
# parse ssGSEA hallmarks scores
f = snakemake.input.CR_hallmarks
hallmarks_all = pd.read_csv(f, sep="\t", header=0, index_col=0).T
# combine both sets of engineered features along CR model ids
CR_combined_test = pd.concat(
    [hallmarks_all, progeny_all], axis=1).dropna(how='any')

# use pre-selected (colinearity, chi2 vs target) features computed
# on the PDX train set
# add any missing feature to CR test set
features_toadd = [
    f for f in PDX_features if f not in CR_combined_test.columns]
CR_combined_test[features_toadd] = np.zeros((len(CR_combined_test),
                                             len(features_toadd)))

X_test = CR_combined_test[PDX_features]
# scale test (0-1)
scaler = MinMaxScaler().fit(X_test)
X_test = pd.DataFrame(scaler.transform(X_test.values),
                      columns=X_test.columns, index=X_test.index)
X_test.\
    to_csv(snakemake.output.preproc_expr,  sep='\t')
