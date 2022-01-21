#!/usr/bin/env python
# coding: utf-8
# Compute CNV features for targetted seq panel genes
#  that map to PDx segmented CN data (from CNVkit)

# ### Imports
# Data manipulation
import pandas as pd
import numpy as np

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 100

target_col = snakemake.params.target_col
# load sample id conversion table, drug response data
f = snakemake.input.response
drug_response_data = pd.read_csv(f, sep="\t")
        
# parse PDx segmented CNV data
f = snakemake.input.cnv
PDx_CNV_data = pd.read_csv(f, sep="\t", index_col=None,
                           names=["chr", 
                           "begin", 
                           "end", 
                           "sample", 
                           "log2R", 
                           "seg_CN", 
                           "depth", 
                           "p_ttest", 
                           "probes", 
                           "weight", 
                           "gene_chr", 
                           "gene_b", 
                           "gene_e", 
                           "gene_symbol",
                           'tumor_types', 
                           "overlapping_admire_segs", 
                           "length_segment-gene_overlap"])
PDx_CNV_data["seg_id"] = PDx_CNV_data.agg(lambda x: f"{x['chr']}:{x['begin']}-{x['end']};{x['sample']}", axis=1)
PDx_CNV_data["gene_HUGO_id"] = PDx_CNV_data["gene_symbol"].str.replace(
    ".", "NA")
PDx_CNV_data["sample"] = PDx_CNV_data["sample"].apply(lambda x:x+"_hum")

# merge with genealogy using Sanger ids 
PDx_CNV_data = pd.merge(PDx_CNV_data,
                       drug_response_data[["sanger_id", "sample_level", 
                       "ircc_id", "is_LMX", "ircc_id_short",
                       "passage",'lineage']].drop_duplicates(),
                       left_on = "sample",
                       right_on = "sanger_id")

# reshape into a gene x model CNV matrix
in_df = PDx_CNV_data[["gene_HUGO_id", "ircc_id", "log2R"]]
# merge multiple CNV values for same gene;model 
in_df = PDx_CNV_data.groupby(["gene_HUGO_id", "ircc_id"]).agg({"log2R" : "mean"}).reset_index()

# discretise log2R into gain, loss, high_gain (aka more than 1.5 copy gain) events
in_df = PDx_CNV_data.groupby(["gene_HUGO_id", "ircc_id"]).agg({"log2R" : "mean"}).reset_index()
loss_thr = snakemake.params.loss_thr
gain_thr = snakemake.params.gain_thr
high_gain_thr = snakemake.params.high_gain_thr
min_CN = round(in_df.log2R.min()) - 1
max_CN = round(in_df.log2R.max()) + 1
in_df["gene_direction"] = pd.cut(in_df.log2R,
                                       bins=[min_CN, loss_thr, 
                                             gain_thr, high_gain_thr, max_CN],
                                       labels=["Loss", "Neutral", "Gain", "highGain"]).astype(str)
# use TCGA compatile labels 
in_df["gene_direction_TCGAcomp"] = in_df["gene_direction"].replace('highGain', 'Gain')
# drop neautral call 
in_df = in_df[~in_df["gene_direction"].str.contains("Neutral")]

# load TCGA CRC CNV data (gene-wise)
# merge with PDx CNV data, this forces TCGA's CNV direction for e/a gene
# CNV event frequency are highly correlated in our PDXs and TCGA CRC
TCGA_CNV_gene_data = pd.read_csv(snakemake.input.TCGA_CNV,
                                 sep="\t", header=0)
TCGA_CNV_gene_data["event_source"] = TCGA_CNV_gene_data.event_id.apply(
    lambda x: "gistic2" if ("Deletion" in x or "Amplification" in x) else "admire1.2")
how_merge = 'inner' # this drops any gene;direction not found in TCGA CRC
if snakemake.params.strict_TCGA_filter == False:
    # this preserves genes not found in the TCGA CRC CNV analysis
    how_merge = 'left'
in_df = pd.merge(in_df,
                        TCGA_CNV_gene_data.dropna(),
                        left_on=["gene_HUGO_id", "gene_direction_TCGAcomp"],
                        right_on=["HUGO_id", "event_direction"],
                        how=how_merge)[["gene_HUGO_id","ircc_id",	
                        "gene_direction", 'log2R']].drop_duplicates().\
                            sort_values("ircc_id").set_index("ircc_id")

in_df['event_name'] = in_df.gene_HUGO_id.astype(str) + "_" + in_df.\
    gene_direction.astype(str)
lo2R_matrix = in_df[['log2R', 'event_name']].reset_index().\
    set_index(["ircc_id", "event_name"]).unstack().dropna(how='all')

# encode gene_dir as binary features,
# account for multiple CNV events for each sample
CNV_matrix = pd.get_dummies(in_df['event_name']).reset_index()\
    .groupby("ircc_id").sum()

# as a baseline 'instability' score simply count the number
# of "Gain" and "Loss" events per sample
CNV_matrix['CN_event_count'] = CNV_matrix.sum(axis=1)

# load drug response data 
ctx3w_cat = drug_response_data[["ircc_id", target_col]].set_index("ircc_id")
# encode target col
Y_class_dict={'PD':0,'SD':1, 'OR':1}
ctx3w_cat[target_col] = ctx3w_cat[target_col].replace(Y_class_dict)

feature_col = CNV_matrix.columns
features_in = pd.merge(ctx3w_cat, CNV_matrix, 
    right_index=True, left_index=True)
# replace na w/t 0
features_in = features_in.fillna(0)
# drop instances w/t missing target 
features_in = features_in[~features_in[target_col].isna()].drop_duplicates()
features_in[feature_col].to_csv(snakemake.output.preproc_CNV,
                   sep='\t')

