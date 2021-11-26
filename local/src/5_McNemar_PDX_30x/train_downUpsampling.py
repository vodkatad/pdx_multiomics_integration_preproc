#!/usr/bin/env python
# coding: utf-8

# # Introduction
# stratified downsampling and upsampling of a training set

# Data manipulation
import math
import pandas as pd
import numpy as np
from numpy.random import default_rng
rng = default_rng()

def is_binary(series, allow_na=False):
    if allow_na:
        series.dropna(inplace=True)
    return sorted(series.unique()) == [0, 1]

# load original train set
f = snakemake.input.X_train 
X_train = pd.read_csv(f, sep='\t', header=0, 
			index_col=0)
# get all categorical/binary col indexes for SMOTENC
categorical_cols = [
	c for c in X_train.columns if '_mut' in c or '_cnv' in c or is_binary(X_train[c])]
all_cols = X_train.columns.tolist()
categorical_cols_idx = [
	all_cols.index(c) for c in categorical_cols] 

target_col = snakemke.params.target_col
f = snakemake.input.response 
response_df = pd.read_csv(f, sep='\t', header=0, 
			index_col=1)
Y_train = response_df.loc[X_train.index].target_col 

upsampling_pct = snakemake.params.upsampling_pct
downsampling_pct = snakemake.params.downsampling_pct
#upsampling_pct = [10,20,50,100]
#downsampling_pct = [90,80,50,25]

# remove a given % of instances in a stratified manner
def remove_pct_models(pct, X, Y, label):
	subset = X.loc[Y[Y == label].index]
	N_toremove = int((pct * len(X)) / (Y.nunique() * 100))
	removed_idx = rng.choice(subset.index.values, N_toremove)
	X_removed = X.loc[removed_idx]
	X_remaining = X.drop(removed_idx)
	Y_removed = Y.loc[removed_idx]
	Y_remaining = Y.drop(removed_idx) 
	return X_removed, X_remaining, Y_removed, Y_remaining

# Synthetic Minority Over-sampling Technique for Nominal and Continuous data
#https://towardsdatascience.com/smote-synthetic-data-augmentation-for-tabular-data-1ce28090debc
def smoteNC(x, y):
    smote = SMOTENC(categorical_features=categorical_cols_idx,
                  sampling_strategy='auto', 
                  #k_neighbors=k_neighbors,
                  random_state=13
                  )
    x, y = smote.fit_resample(x, y)
    return x, y

# perform stratified downsampling at given %
def downsample_dataset(X_train_in, Y_train_in, pct, label='PD'):
	# add PDX model id as a feature to be able to track synthetic models
	X_train_in['ircc_id_int'] = pd.factorize(X_train_in.index)[0]
	# first, balance the dataset
	X_train_in_balanced, Y_train_in_balanced = smoteNC(X_train_in, Y_train_in)
	# stratified downsampling by
	# removing pct / N_classes for each class
	X_train_downsampled = X_train_in_balanced
	Y_train_downsampled = Y_train_in_balanced
	for label in Y_train.unique():
		X_removed, 
		X_remaining, 
		Y_removed, 
		Y_remaining = remove_pct_models(pct, 
						X_train_downsampled, 
						Y_train_downsampled,
						label)
		X_train_downsampled = X_remaining
		Y_train_downsampled = Y_remaining
	N_rem = X_train_in_balanced.shape[0] - X_train_downsampled.shape[0]
	# re-label PDX models using ircc_id, identify synth data
	X_train_downsampled = pd.merge(X_train_in['ircc_id_int'].reset_index(),
					X_train_downsampled,
					on='ircc_id_int').\
					drop('ircc_id_int', axis=1)
	# enumerate synthetic models
	X_train_downsampled = X_train_downsampled.set_index(['ircc_id', 
	X_train_downsampled.groupby('ircc_id').cumcount().astype(str)])\
       					.rename_axis(['ircc_id','count'])
	X_train_downsampled.index = X_train_downsampled.index.map('_'.join)
	Y_train_downsampled.index = X_train_downsampled.index
	print(f"Downsampled by {N_rem/X_train_in_balanced.shape[0] * 100:.3f}%")
	return X_train_downsampled, Y_train_downsampled 

# perform stratified upsampling (balancing) using SMOTENC
def upsample_dataset(X_train_in, Y_train_in, pct, label='PD'):
	# add PDX model id as a feature to be able to track synthetic models
	X_train_in['ircc_id_int'] = pd.factorize(X_train_in.index)[0]
	# first, balance dataset
	X_train_in_balanced, Y_train_in_balanced = smoteNC(X_train_in, Y_train_in)
	# remove N instances, generate N instances by SMOTE balancing
	X_removed, X_remaining, Y_removed, Y_remaining = remove_pct_models(pct, 
								X_train_in_balanced,
								Y_train_in_balanced, label)
	X_train_resampled, Y_train_resampled = smoteNC(X_remaining, Y_remaining)
	# add back N prev removed instances, rebalance dataset
	X_train_upsampled, Y_train_upsampled = smoteNC(pd.concat([X_train_resampled, X_removed]), 
							pd.concat([Y_train_resampled, Y_removed]))
	# re-label PDX models using ircc_id, identify synth data
	X_train_upsampled = pd.merge(X_train_in['ircc_id_int'].reset_index(),
					X_train_upsampled,
					on='ircc_id_int').\
					drop('ircc_id_int', axis=1)
	# enumerate synthetic models
	X_train_upsampled = X_train_upsampled.set_index(['ircc_id', 
	X_train_upsampled.groupby('ircc_id').cumcount().astype(str)])\
       					.rename_axis(['ircc_id','count'])
	X_train_upsampled.index = X_train_upsampled.index.map('_'.join)
	Y_train_upsampled.index = X_train_upsampled.index	
	N_add = X_train_upsampled.shape[0] - X_train_in_balanced.shape[0]
	print(f"Upsampled by {N_add/X_train_in_balanced.shape[0] * 100:.3f}%")
	return X_train_upsampled, Y_train_upsampled

X_up_outfiles = snakemake.output.X_up_outfiles
Y_up_outfiles = snakemake.output.Y_up_outfiles
for trio in zip(upsampling_pct, X_up_outfiles, Y_up_outfiles):
	pctl, X_out, Y_out = trio 
	X_train_upsampled, 
	Y_train_upsampled = upsample_dataset(X_train, Y_train, pctl)
	X_train_upsampled.to_csv(X_out, sep='\t', index=True, header=True)
	Y_train_upsampled.to_csv(Y_out, sep='\t', index=True, header=True)




