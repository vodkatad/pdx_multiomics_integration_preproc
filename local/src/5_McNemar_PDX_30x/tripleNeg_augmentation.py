#!/usr/bin/env python
# coding: utf-8

# # Introduction
# we want to generate additional synthetic training data 
# for clinically relevant subgroups of PDX models:
#   - KRAS-NRAS-BRAF_tripleNeg==1 AND Cetuximab non-responders 
#   - KRAS-NRAS-BRAF_tripleNeg==0 AND Cetuximab responders 

# ### Imports
# Data manipulation
import math
import pandas as pd
import numpy as np

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30

# data augmentation
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE 
from imblearn.over_sampling import ADASYN

def is_binary(series, allow_na=False):
    if allow_na:
        series.dropna(inplace=True)
    return sorted(series.unique()) == [0, 1]

# parse preprocessed training dataset
f = snakemake.input.X_train
X_train = pd.read_csv(f, sep='\t', header=0, 
			index_col=0)
# get all categorical/binary col indexes for SMOTENC
categorical_cols = [c for c in X_train.columns if '_mut' in c or \
    '_cnv' in c or is_binary(X_train[c])]
all_cols = X_train.columns.tolist()
categorical_cols_idx = [
    all_cols.index(c) for c in categorical_cols] 
X_train.head()
X_train.shape

# parse full (train + test) drug response data
f = snakemake.input.response
response_df = pd.read_csv(f, sep='\t', header=0, 
			index_col=1)
response_df.Cetuximab_Standard_3wks_cat.value_counts()
Y_train = response_df.loc[X_train.index].Cetuximab_Standard_3wks_cat 
Y_train.value_counts()

# load triple negative feature
f = snakemake.input.tripleNeg
tripleNeg_df = pd.read_csv(f, sep='\t', header=0, 
			index_col=0)['KRAS_BRAF_NRAS_triple_neg']

# create a temporary target variable for clinical subgroup augmentation
# by combining  Cetuximab response and KRAS-NRAS-BRAF tripleNeg status
logfile = snakemake.log[0]
combined_target = pd.concat([
    response_df.Cetuximab_Standard_3wks_cat,
    tripleNeg_df], axis=1)
combined_target['tripleNeg_response'] = combined_target.\
    Cetuximab_Standard_3wks_cat + 	'_tripleNeg=' +\
         combined_target.KRAS_BRAF_NRAS_triple_neg.\
             astype(int).astype(str)
# slice training PDX models only
tripleNeg_response_Y_train = combined_target.\
    loc[X_train.index].tripleNeg_response
# log original combined categories count
with open(logfile, 'w') as f:
    f.write("The un-augmented training dataset contains: \n")
orig_counts = tripleNeg_response_Y_train.value_counts()
orig_counts.to_csv(logfile, mode='a')

#https://towardsdatascience.com/smote-synthetic-data-augmentation-for-tabular-data-1ce28090debc
def smote(x, y):
    # Synthetic Minority Over-samping Technique
    # 
    # sampling_strategy: determines the portion of samples to 
    #                    generate with respect to the majority class
    # k_neighbors : number of neighbors to be considered for each sample
    smote = SMOTE(sampling_strategy='not majority', 
                  #k_neighbors=k_neighbors,
                  random_state=13
                  )
    x, y = smote.fit_resample(x, y)
    
    return x, y

def smoteNC(x, y):
    # Synthetic Minority Over-sampling Technique for Nominal and Continuous.
    smote = SMOTENC(categorical_features=categorical_cols_idx,
                  sampling_strategy='auto', 
                  #k_neighbors=k_neighbors,
                  random_state=13
                  )
    x, y = smote.fit_resample(x, y)
    return x, y

def bordersmote(x, y):
    # Borderline-SMOTE
    # 
    # sampling_strategy: determines the portion of samples to 
    #                    generate with respect to the majority class
    # k_neighbors : number of neighbors to be considered for each sample
    # m_neighbors : number of neighbors to consider to determine if a sample is danger
    bordersmote = BorderlineSMOTE(sampling_strategy=1, 
                                  #k_neighbors=k_neighbors, 
                                  #m_neighbors=m_neighbors
                                  )
    x, y = bordersmote.fit_resample(x, y)
    return x, y
    
def adasyn(x, y):
    # Adaptive Synthetic
    # 
    # sampling_strategy: determines the portion of samples to 
    #                    generate with respect to the majority class
    # n_neighbors : number of neighbors to be considered for each sample
    adasyn = ADASYN(sampling_strategy=1,
                   #n_neighbors=n_neighbors
                   )
    x, y = adasyn.fit_resample(x, y)
    return x, y

# add factorised PDX model id as a feature 
# to be able to track synthetic models
X_train_in = X_train.copy()
X_train_in['ircc_id_int'] = pd.factorize(X_train_in.index)[0]
X_train_resampled, Y_train_resampled = smoteNC(X_train_in, 
                                        tripleNeg_response_Y_train)
# re-label PDX models using original ircc_id
X_train_resampled = pd.merge(
    X_train_in['ircc_id_int'].reset_index(),
	X_train_resampled,
	on='ircc_id_int').\
		drop('ircc_id_int', axis=1)
# enumerate synthetic models
X_train_resampled = X_train_resampled.set_index(['ircc_id', 
	X_train_resampled.groupby('ircc_id').cumcount().astype(str)])\
       .rename_axis(['ircc_id','count'])
X_train_resampled.index = X_train_resampled.\
    index.map('_'.join)
f = snakemake.output.X_train_resampled
X_train_resampled.to_csv(f, sep='\t', 
                        index=True, 
                        header=True)

# log augmented combined categories count
with open(logfile, 'a') as f:
    f.write("\n\n The augmented training dataset contains: \n")
aug_counts = Y_train_resampled.value_counts()
aug_counts.to_csv(logfile, mode='a')
# split temporary target variable, 
# recover original binary Cetuximab response target
new = Y_train_resampled.str.split("_", n = 1, expand = True)
Y_train_resampled_out = pd.Series(new[0].values, 
	name='Cetuximab_Standard_3wks_cat',
	index=X_train_resampled.index)
f = snakemake.output.Y_train_resampled
Y_train_resampled_out.to_csv(f, sep='\t', 
                        index=True, 
                        header=True)




