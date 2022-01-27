# Data manipulation
import pandas as pd
import numpy as np
from itertools import combinations

# feature scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# feature selection
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2

CR_target_col = snakemake.params.CR_target_col
# load CR response, idTab
f = snakemake.input.CR_idTab
CR_idTab = pd.read_csv(f, sep="\t", header=0)
# format CR id as in mut table
CR_idTab["short_CR_id"] = CR_idTab.sample_ID_in_COSMIC.\
    str.split("_").apply(lambda x: "_".join(x[:3]))
# load CR Cetuximab response
f = snakemake.input.CR_meta
CR_meta = pd.read_csv(f, sep="\t", header=0)

# add CR ids to CR Cetuximab response
# via Sanger Ids
CR_response = pd.merge(
    CR_idTab,
    CR_meta,
    left_on="short_CR_id",
    right_on="id")

# load CR SNP calls,
# merge SNPs, drug response using Sanger ids
f = snakemake.input.CR_mut
CR_mut_data = pd.read_csv(f, sep="\t", header=0)
CR_mut_data["Sample_short"] = CR_mut_data.Sample.\
    str.split("_").apply(lambda x: x[0])
CR_features_pre = pd.merge(CR_response[["PD_ID", "short_CR_id", CR_target_col]].dropna(),
                           CR_mut_data[["Sample_short", "Gene",
                                        "Protein"]].drop_duplicates(),
                           right_on="Sample_short",
                           left_on="PD_ID",
                           how='left').drop(["PD_ID",
                                             "Sample_short",
                                             "PD_ID",
                                             "cetuxi_recist"], axis=1)

# 1-hot encode genes, account for multiple mut in same gene;sample
CR_features_in = pd.get_dummies(CR_features_pre.Gene)
CR_features_in["short_CR_id"] = CR_features_pre.short_CR_id  # add sample ids
CR_features_in = CR_features_in.groupby("short_CR_id").sum()
CR_features_in = CR_features_in.where(
    CR_features_in > 0, other=1)  # just whether gene is mut or not

# load PDX sample id conversion table, drug response data
PDX_drug_response_data = pd.read_csv(
    snakemake.input.PDX_response, sep="\t")
# use only PDX train models
train_models = PDX_drug_response_data[
    PDX_drug_response_data.is_test == False].ircc_id.tolist()
# load driver annotation for PDx models
f = snakemake.input.PDX_mut
driver_data = pd.read_csv(f, "\t", header=0).rename(columns={'Sample':
                                                             'sanger_id'})
drug_mut_df = pd.merge(PDX_drug_response_data, driver_data,
                       on="sanger_id", how="left")

# transform df to have a gene x sample binary mutation matrix
# including all driver genes
PDX_features_pre = drug_mut_df[drug_mut_df["Final_Driver_Annotation"] == True][
    ["ircc_id",
     "Gene"]
].drop_duplicates()
# 1-hot encode genes, vector sum on sample to
PDX_features_in = pd.get_dummies(PDX_features_pre.Gene)
PDX_features_in["ircc_id"] = PDX_features_pre.ircc_id
# account for multiple mut in same sample
PDX_features_in = PDX_features_in.groupby("ircc_id").sum()
PDX_features_in = PDX_features_in.where(
    PDX_features_in > 0, other=1)
PDX_features = PDX_features_in.columns.tolist()
# keep only training set models
train_models = [m for m in train_models if m in PDX_features_in.index]
PDX_features_in = PDX_features_in.loc[train_models]

cols_tokeep = [c for c in PDX_features_in.columns if
               c in CR_features_in.columns]
cols_toadd = [c for c in PDX_features_in.columns if
              c not in CR_features_in.columns]
CR_features_in.shape
# filter CR features using PDX geneset
CR_features_in = CR_features_in[cols_tokeep]
# add missing gene cols as 0s
CR_features_in[cols_toadd] = pd.DataFrame(
    np.zeros((len(CR_features_in),
              len(cols_toadd)), dtype=int),
    columns=cols_toadd)
# reorder features as in PDX df
CR_features_in = CR_features_in[
    PDX_features_in.columns].\
    fillna(0).astype(int)
CR_features = CR_features_in.columns.tolist()

# combine train, test datasets
test_models = CR_features_in.index.tolist()
# combine the PDX Y train set and the CR Y test set
y_test = CR_response[["short_CR_id", "cetuxi_recist"]].\
    set_index("short_CR_id").dropna()
# encode target
CR_class_dict = {'PD': 0, 'SD': 1, 'OR': 1}
y_test = y_test[snakemake.params.CR_target_col].replace(CR_class_dict)
y_train = PDX_drug_response_data[
    ["ircc_id", snakemake.params.PDX_target_col]
].set_index('ircc_id')
CR_class_dict = {'PD': 0, 'OR+SD': 1}
y_train = y_train[
    snakemake.params.PDX_target_col].replace(CR_class_dict)

all_df = pd.concat([PDX_features_in, CR_features_in])
all_df.index = train_models + test_models
# fill any missing feature with 0s
feature_col = all_df.columns.tolist()
all_df = all_df.fillna(0)

y_train = y_train.loc[train_models]
y_test = y_test.loc[test_models]
X_train = all_df.loc[train_models, feature_col]
X_test = all_df.loc[test_models, feature_col]
all_df_scaled = pd.concat([X_train, X_test])

# load pre-computed PDX features
# only select features that have been selected there
f = snakemake.input.PDX_preproc_mut
PDX_preproc_mut = pd.read_csv(f, sep='\t').\
    set_index('ircc_id')
features_tokeep = PDX_preproc_mut.columns


# univariate chi2 to establish a baseline for feature combos
print('selecting top features for crosses')
chi2_stat, pval = [pd.Series(arr) for arr in chi2(X_train, y_train)]
chi2_df = pd.concat([chi2_stat, pval], axis=1)
chi2_df.index = feature_col
chi2_df.columns = ['chi2_stat', 'Pval']
chi2_df = chi2_df.sort_values('chi2_stat', ascending=False)
pctl_tr = chi2_df.chi2_stat.describe().loc['75%']  # set a chi2 pctl threshold
# get top gene features from the training set
# sorted by chi2 stat
top_features = chi2_df[(chi2_df.chi2_stat > pctl_tr)].index

# build new features by counting the unique number of protein mut per gene per sample
pre_PDX = drug_mut_df[drug_mut_df.Gene.isin(feature_col)][
    ["ircc_id",
     "Gene",
     "Protein"]
].drop_duplicates().groupby(["ircc_id", "Gene"]).Protein.nunique().\
    unstack().loc[train_models, :]
pre_CR = CR_features_pre[CR_features_pre.Gene.isin(feature_col)][[
    'short_CR_id',
    'Gene',
    'Protein'
]].drop_duplicates().groupby(["short_CR_id", "Gene"]).Protein.nunique().\
    unstack().loc[test_models, :]
# rename indeces
pre_PDX.index = pre_PDX.index.tolist()
pre_PDX.columns = pre_PDX.columns.tolist()
pre_CR.index = pre_CR.index.tolist()
pre_CR.columns = pre_CR.columns.tolist()
pre = pd.concat([pre_PDX, pre_CR])

# these MultiMut features are 0 when there's 0 or 1 mut in gene, else
# they hold the count of unique mut per gene
multiple_mut = pre.replace({1: np.nan}).dropna(axis=1, how='all').fillna(0)
multiple_mut.columns = [c+"_MultiMut" for c in multiple_mut.columns]
# count tot unique mut burden per sample
multiple_mut["unique_mut_burden"] = pre.apply(pd.Series.sum, axis=1)

# compute 2x and 3x feature crosses (both for train and test sets)
# use only top gene features
# which have been selected on the train (PDX) set
print('computing feture crosses')
interactions2 = list(combinations(top_features, 2))
interactions3 = list(combinations(top_features, 3))
all_df_new = all_df_scaled.copy()
new_features = []
for duo in interactions2:
    f1, f2 = duo
    v = all_df_new[f1] * all_df_new[f2]
    k = f"{f1}_{f2}_double_pos"
    all_df_new[k] = v
    new_features.append(k)
for trio in interactions3:
    f1, f2, f3 = trio
    v = (all_df_new[f1] + all_df_new[f2] + all_df_new[f3]
         ).replace({0: 1, 1: 0, 2: 0, 3: 0})
    k = f"{f1}_{f2}_{f3}_triple_neg"
    all_df_new[k] = v
    new_features.append(k)
    v = (all_df_new[f1] * all_df_new[f2] * all_df_new[f3])
    k = f"{f1}_{f2}_{f3}_triple_pos"
    all_df_new[k] = v
    new_features.append(k)

# add MultiMut features
# aka are there multiple muts on this gene in this sample?
new_features.extend(multiple_mut.columns)
all_df_new = pd.merge(all_df_new, multiple_mut,
                      left_index=True,
                      right_index=True,
                      how="left")
all_df_new_scaled = all_df_new.fillna(0)
# rescale all non-binary new features
nonbin_features = [c for c in all_df_new if all_df_new[c].nunique() > 2]
all_df_new_scaled[nonbin_features] = MinMaxScaler().fit_transform(
    all_df_new[nonbin_features].values)

# build a new training, test dataset including all new features
X_train_new = all_df_new_scaled.loc[train_models, feature_col + new_features]
X_test_new = all_df_new_scaled.loc[test_models, feature_col + new_features]
chi2_stat, pval = [pd.Series(arr) for arr in chi2(X_train_new.values, y_train)]
chi2_df = pd.concat([chi2_stat, pval], axis=1)
chi2_df.index = X_train_new.columns
chi2_df.columns = ['chi2_stat', 'Pval']
chi2_df['Padj'] = chi2_df.Pval + len(chi2_df)
chi2_df = chi2_df.sort_values('chi2_stat', ascending=False)

# since there are multiple crosses containing each gene name,
# pick the best 3x and 2x cross involving said gene
features_tokeep = feature_col
chi2_new_df = chi2_df.copy()
for gene in reversed(top_features.tolist()):  # inverse rank by chi2
    # pick the best (chi2) feature duo involving gene
    gene_duos = chi2_new_df[(chi2_new_df.index.str.contains(gene)) &
                            (chi2_new_df.index.str.contains('_double_'))]
    try:
        best_duo = gene_duos.index[0]
    except IndexError:
        continue
    duos_todrop = gene_duos.index[1:].tolist()  # drop the others
    # pick best trio involving gene
    gene_trios = chi2_new_df[(chi2_new_df.index.str.contains(gene)) &
                             (chi2_new_df.index.str.contains('_triple_'))]
    try:
        best_trio = gene_trios.index[0]
    except IndexError:
        continue
    trios_todrop = gene_trios.index[1:].tolist()
    # drop unselected features
    chi2_new_df = chi2_new_df.drop(duos_todrop + trios_todrop)
features_tokeep = chi2_new_df.index.tolist()

# these should be a subset of the features generated here since both
# sets are engineered from the same PDX train set replicate
features_toadd = [f for f in features_tokeep if f not in all_df_new_scaled]
all_df_new_scaled[features_toadd] = np.zeros(
    (len(all_df_new_scaled), len(features_toadd)))
f = snakemake.output.preproc_mut
all_df_new_scaled[features_tokeep].\
    fillna(0).to_csv(f, sep='\t')
