#!/usr/bin/env python
# coding: utf-8

# # Introduction
# stratified downsampling and upsampling of a training set

# Data manipulation
import math
import pandas as pd
import numpy as np
from numpy.random import default_rng
from imblearn.over_sampling import SMOTENC
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

target_col = snakemake.params.target_col
f = snakemake.input.response 
response_df = pd.read_csv(f, sep='\t', header=0).set_index('ircc_id')
Y_train = response_df.loc[X_train.index, target_col]
logfile = snakemake.log[0]

def remove_pct_overall(pct, X, Y):
	N_toremove = int((pct * len(X)) / 100)
	N_classes = Y.nunique()
	classes = Y.unique()
	N_toremove_perClass = int(N_toremove / N_classes) 
	removed_idxs = []
	for C in classes:
		# slice subeset of models labelled as class
		subset = X.loc[Y[Y == C].index.values]
		removed_idxs.extend(
			rng.choice(subset.index.values, 
			N_toremove_perClass))
	X_removed = X.loc[removed_idxs]
	X_remaining = X.drop(removed_idxs)
	Y_removed = Y.loc[removed_idxs]	
	Y_remaining = Y.drop(removed_idxs)
	return X_removed, X_remaining, Y_removed, Y_remaining

# remove a given % of label-specific instances in a stratified manner
def remove_pct_perLabel(pct, X, Y, label):
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
	X_removed, X_train_downsampled, Y_removed, Y_train_downsampled = remove_pct_overall(pct, 
		X_train_in_balanced, 
		Y_train_in_balanced)

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
	with open(logfile, 'a') as f:
		f.write(f"Downsampled by {N_rem/X_train_in_balanced.shape[0] * 100:.3f}% from {X_train_in_balanced.shape[0]}\n")
	return X_train_downsampled, Y_train_downsampled 

# perform stratified upsampling (balancing) using SMOTENC
def upsample_dataset(X_train_in, Y_train_in, pct, label='PD'):
	# add PDX model id as a feature to be able to track synthetic models
	X_train_in['ircc_id_int'] = pd.factorize(X_train_in.index)[0]
	# first, balance dataset
	X_train_in_balanced, Y_train_in_balanced = smoteNC(X_train_in, Y_train_in)
	# remove N instances, generate N instances by SMOTE balancing
	X_removed, X_remaining, Y_removed, Y_remaining = remove_pct_perLabel(pct, 
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
	with open(logfile, 'a') as f:
		f.write(f"Upsampled by {N_add/X_train_in_balanced.shape[0] * 100:.3f}% from {X_train_in_balanced.shape[0]}\n")
	return X_train_upsampled, Y_train_upsampled

X_outfile = snakemake.output.X_outfile
Y_outfile = snakemake.output.Y_outfile
direction = snakemake.params.direction
pct = int(snakemake.params.pct)
if direction == 'up':
	X_train_resampled, Y_train_resampled = upsample_dataset(X_train, Y_train, pct)
else:
	X_train_resampled, Y_train_resampled = downsample_dataset(X_train, Y_train, pct)	
X_train_resampled.to_csv(X_outfile, sep='\t', index=True, header=True)
Y_train_resampled.to_csv(Y_outfile, sep='\t', index=True, header=True)





