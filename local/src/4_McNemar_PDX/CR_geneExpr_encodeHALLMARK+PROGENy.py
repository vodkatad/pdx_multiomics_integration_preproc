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
Y_PDX = drug_response_data[["ircc_id_short", "ircc_id",
                            PDX_target_col, "is_test"]].\
    set_index("ircc_id_short")
PDX_train_models = Y_PDX[Y_PDX.is_test == False].dropna(how='any').\
    ircc_id.tolist()

# load the progeny scores for PDX test set models
progeny_test = pd.read_csv(snakemake.input.PDX_progeny_test, sep='\t')
# load the progeny scores for PDX train set models
progeny_train = pd.read_csv(snakemake.input.PDX_progeny_train, sep='\t')
# concat
#progeny_train = pd.concat([progeny_test, progeny_train])
progeny_train.columns = ['ircc_id'] + ['PROGENy_' +
                                       c for c in progeny_train.columns[1:].tolist()]
progeny_train = progeny_train.set_index('ircc_id')
progeny_features = progeny_train.columns.tolist()

# parse PDX test, train hallmark ssGSEA scores and concat
f = snakemake.input.PDX_hallmarks_test
hallmarks_test = pd.read_csv(f, sep="\t", header=0, index_col=0).T
f = snakemake.input.PDX_hallmarks_train
hallmarks_train = pd.read_csv(f, sep="\t", header=0, index_col=0).T
# concat
#hallmarks_train = pd.concat([hallmarks_test, hallmarks_train])
# index on ircc_short
hallmarks_train.index = [i[:7] for i in hallmarks_train.index.tolist()]
# join on ircc_id, aggregate RNAseq replicates by mean, set ircc_id as index
hallmarks_train = pd.merge(
    hallmarks_train,
    Y_PDX[["ircc_id", PDX_target_col]],
    left_index=True, right_index=True)
hallmarks_train = hallmarks_train.groupby("ircc_id").apply(
    np.mean).reset_index().set_index('ircc_id')
hallmarks_features = hallmarks_train.columns.tolist()
# combine PROGENy, halmarks scores along ircc_id
PDX_combined_train = pd.concat(
    [hallmarks_train, progeny_train], axis=1).dropna(how='any')
PDX_train_models = [m for m in PDX_train_models if
                    m in PDX_combined_train.index]


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

# compute PDX-CR engineered features intersection set
shared_features = PDX_combined_train.columns.tolist()
# add any missing feature to CR
features_toadd = [
    f for f in shared_features if f not in CR_combined_test.columns]
CR_combined_test[features_toadd] = np.zeros((len(CR_combined_test),
                                             len(features_toadd)))
X_test = CR_combined_test[shared_features]
X_train = PDX_combined_train.loc[PDX_train_models, shared_features]
# encode target
CR_class_dict = {'PD': 0, 'SD': 1, 'OR': 1}
y_test = y_test[CR_target_col].replace(CR_class_dict)
PDX_class_dict = {'PD': 0, 'OR+SD': 1}
y_train = Y_PDX.reset_index().set_index('ircc_id')[
    PDX_target_col].loc[PDX_train_models].\
    replace(PDX_class_dict)
# scale train, test (0-1) separately
scaler = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train.values),
                       columns=X_train.columns, index=X_train.index)
scaler = MinMaxScaler().fit(X_test)
X_test = pd.DataFrame(scaler.transform(X_test.values),
                      columns=X_test.columns, index=X_test.index)

print([X_test.shape, y_test.shape, X_train.shape, y_train.shape])
# compute feature corr matrix on PDX features
corr_thrs = .7
corr_train_df = X_train.corr().stack().reset_index()
corr_train_df.columns = ['F1', 'F2', 'pcc']
# remove self comparisons, sort
corr_train_df['abs_pcc'] = np.abs(corr_train_df.pcc)
corr_train_df = corr_train_df[corr_train_df.F1 !=
                              corr_train_df.F2].\
    sort_values('pcc', ascending=False)
# get all features w/t at least one abs(corr) > corr_thrs
colinear_df = corr_train_df[corr_train_df.abs_pcc > corr_thrs].\
    sort_values('abs_pcc', ascending=False)
# compute chi2 vs target for all colinear features
colinear_features = set(colinear_df.F1.tolist() +
                        colinear_df.F2.tolist())
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
        if C1 > C2 and F1 not in features_todrop:
            features_todrop.append(F2)
            log.write(f"PCC: {pcc}; keep {F1} ({C1}), drop {F2} ({C2})\n")
        elif C1 < C2 and F2 not in features_todrop:
            features_todrop.append(F1)
            log.write(f"PCC: {pcc}; keep {F2} ({C2}), drop {F1} ({C1})\n")

chi2_df.loc[list(set(features_todrop))].\
    sort_values('chi2_stat', ascending=False)
print(len(features_todrop))
# save stacked model input including train and test sets (non-scaled)
pd.concat([X_train, X_test]).\
    drop(list(set(features_todrop)), axis=1).\
    to_csv(snakemake.output.preproc_expr,  sep='\t')
