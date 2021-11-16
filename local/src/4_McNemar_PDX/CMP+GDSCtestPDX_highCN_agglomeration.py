#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Compute CNV stats for intogen genes that map to PDx segmented CN data (from CNVkit)

# ### Imports
# Data manipulation
import pandas as pd
import numpy as np

# load PDX ids, train/test split, response
f = snakemake.input.PDX_response
PDX_target_col = snakemake.p[arams.PDX_target_col] 
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode Cetuximab response as binary bar
Y_class_dict={'PD':0,'SD':1, 'OR':1}
Y[PDX_target_col] = Y[PDX_target_col].replace(Y_class_dict)
PDX_test_models = Y[Y.is_test == True].index.unique()

# load preprocessed PDX CN features (includes "high_CN" features)
f = snakemake.input.PDX_cnv_pre
PDX_features_in = pd.read_csv(f, sep='\t', index_col=0).\
    drop(PDX_target_col, axis=1)
PDX_test_models = [m for m in PDX_test_models if m in PDX_features_in.index]
PDX_X_test = PDX_features_in.loc[PDX_test_models, :]
PDX_y_test = Y.loc[PDX_test_models, PDX_target_col]

# load CMP CNV, drop heaer lines, rename id col
f = snakemake.input.CMP_cnv
CMP_CNV = pd.read_csv(f).drop([0, 1]).drop('model_id', axis=1)
CMP_CNV = CMP_CNV.rename(columns={'Unnamed: 1' : 'gene_symbol'})
# load TCGA CNV genewise
f = snakemake.input.TCGA_CNV
TCGA_CNV_gene_data = pd.read_csv(f,sep="\t", header=0)
TCGA_CNV_gene_data["event_source"] = TCGA_CNV_gene_data.event_id.apply(
    lambda x: "gistic2" if ("Deletion" in x or "Amplification" in x) else "admire1.2")
# load the GDSC drug response data
f = snakemake.input.CMP_response
drug_response_data = pd.read_csv(f, 
sep="\t", index_col=None)
all_models = drug_response_data.SANGER_MODEL_ID.values
CMP_target_col = snakemake.params.CMP_target_col
# slice cell line models
CMP_CNV = CMP_CNV[[c for c in all_models if c in CMP_CNV.columns] + ['gene_symbol']]
# reshape
CMP_CNV = CMP_CNV.set_index('gene_symbol').stack().reset_index()
CMP_CNV.columns = ['gene_symbol', 'model_id', 'CNV']
# bin into direction labels
loss_thr = -.5 # not log2R so just cut on integer CN values
gain_thr = .5
# use log2R>.75 in PDX to define "highGain" (aka more than 1.5 copies gained)
high_gain_thr = 1.5
min_CN = -3
max_CN = 3
CMP_CNV["gene_direction"] = pd.cut(CMP_CNV.CNV.astype(float),
                                       bins=[min_CN, loss_thr, 
                                             gain_thr, high_gain_thr, max_CN],
                                       labels=["Loss", "Neutral", "Gain", "highGain"]).astype(str)
CMP_CNV["gene_direction_TCGAcomp"] = CMP_CNV["gene_direction"].replace('highGain', 
                                                                        'Gain') # use TCGA compatile labels
# merge with TCGA CNV data, this forces TCGA's CNV direction for e/a gene
CMP_CNV = pd.merge(CMP_CNV,
                        TCGA_CNV_gene_data.dropna(),
                        left_on=["gene_symbol", "gene_direction_TCGAcomp"],
                        right_on=["HUGO_id", "event_direction"])[["gene_symbol","model_id",
                        "gene_direction"]].drop_duplicates().\
                            sort_values("model_id").set_index("model_id")

CMP_CNV['event_name'] = CMP_CNV.gene_symbol.astype(
    str) + "_" + CMP_CNV.gene_direction.astype(str)
# encode gene_dir as binary features,
# account for multiple CNV events for each sample
CNV_matrix = pd.get_dummies(
    CMP_CNV['event_name']).reset_index().groupby("model_id").sum()

# ### Combine PDX test set, CMP+GDSC training set
# Feature set intersection
shared_features = [f for f in PDX_features_in.columns if f in CNV_matrix.columns]
PDX_X_test = PDX_features_in.loc[PDX_test_models, shared_features]
CMPGDSC_all_df = CNV_matrix[shared_features]
# slice train, test models  
TT_df = drug_response_data[drug_response_data.SANGER_MODEL_ID.isin(CMPGDSC_all_df.index)][
    ['SANGER_MODEL_ID', 'is_test']]                    
# load GDSC drug response data
CMPGDSC_target_col = 'IC50_byMedian_cat'
Y = drug_response_data[['SANGER_MODEL_ID', 'is_test', CMPGDSC_target_col]].\
    set_index('SANGER_MODEL_ID')
feature_col = [c for c in CMPGDSC_all_df.columns if c != CMPGDSC_target_col]
# drop duplicated instances (model_id) from index
CMPGDSC_all_df = CMPGDSC_all_df[~CMPGDSC_all_df.index.duplicated(keep='first')]
# use the full CMP-GDSC dataset as a training set
CMPGDSC_train_models = TT_df.SANGER_MODEL_ID.unique()
CMPGDSC_X_train = CMPGDSC_all_df.loc[CMPGDSC_train_models, feature_col]
CMPGDSC_y_train  = Y.loc[CMPGDSC_train_models, CMPGDSC_target_col]

# fit scaler, scale 
CMPGDSC_X_train = pd.DataFrame(MinMaxScaler().fit_transform(CMPGDSC_X_train.values),
                            columns=CMPGDSC_X_train.columns,
                            index=CMPGDSC_X_train.index)
# rename indeces, concat, save to file
PDX_X_test.index = PDX_X_test.index.tolist()
CMPGDSC_X_train.index = CMPGDSC_X_train.index.tolist()
f = snakemake.output.preproc_CNV
pd.concat([PDX_X_test, CMPGDSC_X_train]).to_csv(f, sep='\t')
