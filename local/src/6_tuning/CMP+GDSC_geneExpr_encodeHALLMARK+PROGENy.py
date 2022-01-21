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

# ### Load PDX PROGENY, Hallmarks scores
# load sample id conversion table, drug response data
PDX_target_col = snakemake.params.PDX_target_col 
drug_response_data = pd.read_csv(snakemake.input.PDX_response,
                                 sep="\t")
# index Y on ircc_short to map to RNAseq ids
Y = drug_response_data[["ircc_id_short", "ircc_id",
                        PDX_target_col, "is_test"]].\
                            set_index("ircc_id_short")
PDX_test_models = Y[Y.is_test == True].ircc_id.tolist()
y_test = Y[Y.ircc_id.isin(PDX_test_models)][PDX_target_col]
# load the progeny scores for PDX test set models
progeny_test = pd.read_csv(snakemake.input.PDX_progeny_test, sep='\t')
# load the progeny scores for PDX train set models
progeny_train = pd.read_csv(snakemake.input.PDX_progeny_train, sep='\t')
# concat
progeny_test = pd.concat([progeny_test, progeny_train])
progeny_test.columns = ['ircc_id'] + ['PROGENy_' +
                                      c for c in progeny_test.columns[1:].tolist()]
progeny_test = progeny_test.set_index('ircc_id')
progeny_features = progeny_test.columns.tolist()

# parse PDX test, train hallmark ssGSEA scores and concat
f = snakemake.input.PDX_hallmarks_test
hallmarks_test = pd.read_csv(f, sep="\t", header=0, index_col=0).T
f = snakemake.input.PDX_hallmarks_train
hallmarks_train = pd.read_csv(f, sep="\t", header=0, index_col=0).T
hallmarks_test = pd.concat([hallmarks_test, hallmarks_train])
# index on ircc_short
hallmarks_test.index = [i[:7] for i in hallmarks_test.index.tolist()]
# join on ircc_id, aggregate RNAseq replicates by mean, set ircc_id as index
hallmarks_test = pd.merge(
    hallmarks_test, Y[["ircc_id", PDX_target_col]], left_index=True, right_index=True)
hallmarks_test = hallmarks_test.groupby("ircc_id").apply(
    np.mean).reset_index().set_index('ircc_id')
hallmarks_features = hallmarks_test.columns.tolist()
# combine PROGENy, halmarks scores along ircc_id
PDX_combined_test = pd.concat(
    [hallmarks_test, progeny_test], axis=1).dropna(how='any')

# ### Load and combine CMP PROGENy and MSdb Hallmarks features
# These have been computed over the full CMP set
# load GDSC Cetuximab response data
f = snakemake.input.CMP_response
drug_response_data = pd.read_csv(f,
                                 sep="\t", index_col=None)
all_models = drug_response_data.SANGER_MODEL_ID.values
CMP_target_col = snakemake.params.CMP_target_col
Y = drug_response_data[['SANGER_MODEL_ID',
                        CMP_target_col]].\
                            set_index('SANGER_MODEL_ID')
# use the entire CMP dataset as train
y_train = Y.loc[:, CMP_target_col]

# parse CMP PROGENy scores for all instances
progeny_all = pd.read_csv(snakemake.input.CMP_progeny,
                          header=0, index_col=0)
# add column prefix
progeny_all.columns = ['PROGENy_' + c for c in progeny_all.columns]
# parse ssGSEA hallmarks scores
f = snakemake.input.CMP_hallmarks
hallmarks_all = pd.read_csv(f, sep="\t", header=0, index_col=0).T
# combine both sets of engineered features along CMP model ids
CMPGDSC_combined_train = pd.concat(
    [hallmarks_all, progeny_all], axis=1).dropna(how='any')

# compute PDX-CMP engineered features intersection set
shared_features = [
    f for f in CMPGDSC_combined_train.columns if f in PDX_combined_test.columns]
X_train = CMPGDSC_combined_train[shared_features]
X_test = PDX_combined_test[shared_features]

# scale train, test (0-1) separately
scaler = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train.values),
                       columns=X_train.columns, index=X_train.index)
scaler = MinMaxScaler().fit(X_test)
X_test = pd.DataFrame(scaler.transform(X_test.values),
                      columns=X_test.columns, index=X_test.index)

# compute feature corr matrix on CMP features
corr_thrs = .7
corr_train_df = X_train.corr().stack().reset_index()
corr_train_df.columns = ['F1', 'F2', 'pcc']
# remove self comparisons, sort
corr_train_df['abs_pcc'] = np.abs(corr_train_df.pcc)
corr_train_df = corr_train_df[corr_train_df.F1 !=
                              corr_train_df.F2].	sort_values('pcc', ascending=False)
# get all features w/t at least one abs(corr) > corr_thrs
colinear_df = corr_train_df[corr_train_df.abs_pcc > corr_thrs].\
    sort_values('abs_pcc', ascending=False)
# compute chi2 vs target for all colinear features
colinear_features = set(colinear_df.F1.tolist() + colinear_df.F2.tolist())
chi2_stat, pval = [pd.Series(arr) for arr in chi2(
    X_train[colinear_features], y_train)]
chi2_df = pd.concat([chi2_stat, pval], axis=1)
chi2_df.index = colinear_features
chi2_df.columns = ['chi2_stat', 'Pval']
chi2_df = chi2_df.sort_values('chi2_stat', ascending=False)
# for each pair of colinear features in descending corr order,
#  keep the one with the highest chi2 stat
features_todrop = []
with open(logfile, "w") as log:
    for F1, F2, pcc in zip(colinear_df.F1.tolist(), colinear_df.F2.tolist(), colinear_df.pcc.tolist()):
        C1 = chi2_df.chi2_stat.loc[F1]
        C2 = chi2_df.chi2_stat.loc[F2]
        if C1 > C2  and F1 not in features_todrop:
            features_todrop.append(F2)
            log.write(f"PCC: {pcc}; keep {F1} ({C1}), drop {F2} ({C2})\n")
        elif C1 < C2 and F2 not in features_todrop:
            features_todrop.append(F1)
            log.write(f"PCC: {pcc}; keep {F2} ({C2}), drop {F1} ({C1})\n")

chi2_df.loc[list(set(features_todrop))].\
    sort_values('chi2_stat', ascending=False)
# save stacked model input including train and test sets (non-scaled)
pd.concat([X_train, X_test]).\
    drop(list(set(features_todrop)), axis=1).\
    to_csv(snakemake.output.preproc_expr,  sep='\t')
