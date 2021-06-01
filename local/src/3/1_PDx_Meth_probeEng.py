#!/usr/bin/env python
# coding: utf-8

# Data manipulation
import pandas as pd
import numpy as np
import vaex
import pickle

# read methylation values for each probe
master_filename = snakemake.input.meth_m
features_in = vaex.open(master_filename)
nrows, ncols = features_in.shape

# read beast probe per gene by sd (higher better)
f = snakemake.input.meth_bestSDs
collapsed_ALLprobes = pd.read_csv(
    f, sep="\t", header=0, index_col=0).reset_index()
collapsed_ALLprobes.head()

# read methylation bionomial test pvalue
f = snakemake.input.meth_B_DTpval
bDTpval = pd.read_csv(f, sep="\t").reset_index()
bDTpval.columns = ["probe", "beta_DT-pvalue"]
bDTpval = bDTpval.set_index("probe")
# read methylation variance for each probe
f = snakemake.input.meth_m_sd
Msd = pd.read_csv(f, sep="\t").reset_index()
Msd.columns = ["probe", "M_sd"]
Msd = Msd.set_index("probe")
probe_stats = pd.concat([Msd, bDTpval], axis=1)

# keep probes w/t binomial FDR < thrs
FDR = float(snakemake.params.binomial_FDR_thrs)
probes_tokeep = pd.Series(probe_stats[(
    probe_stats["beta_DT-pvalue"] < FDR)  # null hypothesis: M distib is unimodal
].index.tolist())

# keep only probes that are the 'best probe' (highest SD) for a given gene
probe_gene = collapsed_ALLprobes[collapsed_ALLprobes.probe.
                                 isin(probes_tokeep)]
probe_gene["feature"] = collapsed_ALLprobes["index"] + \
    '_' + collapsed_ALLprobes.probe
probes_tokeep = probe_gene.probe.unique()

# build dict mapping probe to feature name which includes gene name
probe2feature = dict(zip(probe_gene.probe, probe_gene.feature))

# load sample id conversion table, drug response data
drug_response_data = pd.read_csv(
    snakemake.input.response, sep="\t")

# filter on probes, models
samples_tokeep = drug_response_data.ircc_id.apply(
    lambda x: x.replace("TUM", " ").split()[0]).unique()
samples_tokeep = [c for c in samples_tokeep if c in list(
    features_in.columns.keys())[1:]]
features_clean = features_in[features_in["probe"].isin(
    probes_tokeep)][samples_tokeep + ["probe"]]

# map probes to gene names
features_clean['probe'] = features_clean['probe'].apply(
    lambda x: probe2feature[x])

# convert to pandas, reshape, add target
features_clean_df = features_clean.to_pandas_df()
features_clean_df = features_clean_df.set_index("probe").T
features_clean_df.columns = features_clean_df.columns.tolist()
features_clean_df["ircc_id_short"] = [x[0:7] for x in features_clean_df.index]

target_col = snakemake.params.target_col
features_clean_df = pd.merge(drug_response_data[[
    target_col, "ircc_id_short", "ircc_id", "is_test"]],
    features_clean_df,
    on="ircc_id_short")
# encode target
Y_class_dict = {'PD': 0, 'SD': 1, 'OR': 1}
features_clean_df[target_col] = features_clean_df[target_col].replace(
    Y_class_dict)

train_models = features_clean_df[features_clean_df.is_test ==
                                 False].ircc_id.unique()
test_models = features_clean_df[features_clean_df.is_test ==
                                True].ircc_id.unique()
features_clean_df = features_clean_df.drop(
    ["is_test", "ircc_id_short"], axis=1).set_index("ircc_id")
features_clean_df.to_csv(nakemake.input.preproc_meth, sep="\t")
