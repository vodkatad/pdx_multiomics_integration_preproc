
# ### Imports
# Import libraries and write settings here.

# Data manipulation
import pandas as pd
import numpy as np

# scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# processing
from sklearn.preprocessing import label_binarize, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import ColumnSelector
from sklearn import model_selection

# feature selection
from sklearn.feature_selection import SelectFdr, f_classif, SelectKBest, SelectFromModel, VarianceThreshold, chi2

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30


logfile = snakemake.log[0]

# load sample id conversion table, drug response data
target_col = snakemake.params.target_col 
# load sample id conversion table, drug response data
f = snakemake.input.response 
drug_response_data = pd.read_csv(f,
                                 sep="\t")
Y = drug_response_data[["ircc_id_short", "ircc_id", target_col, "is_test"]].\
    set_index("ircc_id_short")

# parse PROGENy scores for train, test set
f = snakemake.input.progeny_train
progeny_train = pd.read_csv(f)
progeny_train.columns = ['ircc_id'] + ['PROGENy_' + c for c in progeny_train.columns[1:]]
progeny_train = progeny_train.set_index('ircc_id')
f = snakemake.input.progeny_test
progeny_test = pd.read_csv(f)
progeny_test.columns = ['ircc_id'] + ['PROGENy_' + c for c in progeny_test.columns[1:]]
progeny_test = progeny_test.set_index('ircc_id')
progeny_features = list(set(progeny_test.columns.tolist() +  progeny_train.columns.tolist()))

# parse train hallmark ssGSEA
f = snakemake.input.hallmarks_train
hallmarks_train = pd.read_csv(f, sep="\t", header=0, index_col=0).T
# index on ircc_short
hallmarks_train.index = [i[:7] for i in hallmarks_train.index.tolist()] 
# join with ircc_long, aggregate RNAseq replicates by mean, , set ircc_id as index
hallmarks_train = pd.merge(hallmarks_train, Y[["ircc_id", target_col]], \
    left_index=True, right_index=True)
hallmarks_train = hallmarks_train.groupby("ircc_id").\
    apply(np.mean).reset_index().set_index('ircc_id')

y_train = pd.merge(hallmarks_train, Y[["ircc_id", target_col]].set_index('ircc_id'),\
     left_index=True, right_index=True)[target_col]

# parse test hallmark ssGSEA
f = snakemake.input.hallmarks_test
hallmarks_test = pd.read_csv(f, sep="\t", header=0, index_col=0).T
# index on ircc_short
hallmarks_test.index = [i[:7] for i in hallmarks_test.index.tolist()] 
# join with ircc_id, aggregate RNAseq replicates by mean, set ircc_id as index
hallmarks_test = pd.merge(hallmarks_test, Y[["ircc_id", target_col]], \
    left_index=True, right_index=True)
hallmarks_test = hallmarks_test.groupby("ircc_id").\
    apply(np.mean).reset_index().set_index('ircc_id')
y_test = pd.merge(hallmarks_test, Y[["ircc_id", target_col]].set_index('ircc_id'),\
     left_index=True, right_index=True)[target_col]

# combine PROGENy, halmarks scores along ircc_id
combined_train = pd.concat([hallmarks_train, progeny_train], axis=1).dropna(how='any')
combined_test = pd.concat([hallmarks_test, progeny_test], axis=1).dropna(how='any')

# scale train, test (0-1) separately
scaler = MinMaxScaler().fit(combined_train)
scaled_combined_train = pd.DataFrame(scaler.transform(combined_train.values),
          columns=combined_train.columns, index=combined_train.index)
scaler = MinMaxScaler().fit(combined_test)              
scaled_combined_test = pd.DataFrame(scaler.transform(combined_test.values),
          columns=combined_test.columns, index=combined_test.index)    

# compute feature corr matrix on train
corr_thrs = float(snakemake.params.corr_thrs)
corr_train_df = scaled_combined_train.corr().stack().reset_index()
corr_train_df.columns = ['F1', 'F2', 'pcc']
# remove self comparisons, sort
corr_train_df['abs_pcc'] = np.abs(corr_train_df.pcc)
corr_train_df = corr_train_df[corr_train_df.F1 != corr_train_df.F2].\
    sort_values('pcc', ascending=False)

# get all features w/t at least one abs(corr) > corr_thrs
colinear_df = corr_train_df[corr_train_df.abs_pcc > corr_thrs].\
    sort_values('abs_pcc', ascending=False)

# compute chi2 vs target for all colinear features
colinear_features = set(colinear_df.F1.tolist() + colinear_df.F2.tolist()) 
len(colinear_features)
chi2_stat, pval = [pd.Series(arr) for arr in chi2(scaled_combined_train[colinear_features], y_train)]
chi2_df = pd.concat([chi2_stat,pval], axis=1)
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
            log.write(f"PCC: {pcc}; keep {F2} ({C2}),, drop {F1} ({C1})\n")

# save stacked model input
pd.concat([combined_train, combined_test]).\
    drop(list(set(features_todrop)), axis=1).\
        to_csv(snakemake.output.preproc_expr,  sep='\t')
