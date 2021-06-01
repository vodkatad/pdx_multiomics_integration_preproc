#!/usr/bin/env python
# coding: utf-8

#
# Compute CNV stats for intogen genes that map to PDx segmented CN data (from CNVkit)


# Data manipulation
import pandas as pd
import numpy as np

target_col = snakemake.params.target_col
# load sample id conversion table, drug response data
drug_response_data = pd.read_csv(snakemake.input.response,
                                 sep="\t")
# parse PDx segmented CNV data
PDx_CNV_data = pd.read_csv(snakemake.input.cnv,
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
PDx_CNV_data["seg_id"] = PDx_CNV_data.agg(
    lambda x: f"{x['chr']}:{x['begin']}-{x['end']};{x['sample']}", axis=1)
PDx_CNV_data["gene_HUGO_id"] = PDx_CNV_data["gene_symbol"].str.replace(
    ".", "NA")
PDx_CNV_data["sample"] = PDx_CNV_data["sample"].apply(lambda x: x+"_hum")

# load shared set of genes for intogen and targeted sequencing
common_geneset = pd.read_table(snakemake.input.targeted,
                               header=None, sep="\t")
common_geneset = common_geneset[0].tolist()

# merge with genealogy using Sanger ids
PDx_CNV_data = pd.merge(PDx_CNV_data,
                        drug_response_data[["sanger_id", "sample_level",
                                            "ircc_id", "is_LMX", "ircc_id_short",
                                            "passage", 'lineage']].drop_duplicates(),
                        left_on="sample",
                        right_on="sanger_id")

# reshape into a gene x model CNV matrix
in_df = PDx_CNV_data[["gene_HUGO_id", "ircc_id", "log2R"]]
# merge multiple CNV values for same gene;model
in_df = PDx_CNV_data.groupby(["gene_HUGO_id", "ircc_id"]).agg({
    "log2R": "mean"}).reset_index()
in_df.head()

# discretise LogR values into "Loss" and "Gain" labels
# using gistic thresholds for TCGA CRC CNV LogR
in_df = PDx_CNV_data.groupby(["gene_HUGO_id", "ircc_id"]).agg({
    "log2R": "mean"}).reset_index()
loss_thr = -.2
gain_thr = .1
min_CN = round(in_df.log2R.min()) - 1
max_CN = round(in_df.log2R.max()) + 1
in_df["gene_direction"] = pd.cut(in_df.log2R,
                                 bins=[min_CN, loss_thr,
                                       gain_thr, max_CN],
                                 labels=["Loss", "Neutral", "Gain"]).astype(str)
# drop neautral calls
in_df = in_df[~in_df["gene_direction"].str.contains("Neutral")]

# load TCGA CNV gene stats
TCGA_CNV_gene_data = pd.read_csv(snakemake.input.TCGA_CNV,
                                 sep="\t", header=0)
TCGA_CNV_gene_data["event_source"] = TCGA_CNV_gene_data.event_id.apply(
    lambda x: "gistic2" if ("Deletion" in x or "Amplification" in x) else "admire1.2")
# merge with PDx CNV data, this forces TCGA's CNV direction for e/a gene
in_df = pd.merge(in_df,
                 TCGA_CNV_gene_data.dropna(),
                 left_on=["gene_HUGO_id", "gene_direction"],
                 right_on=["HUGO_id", "event_direction"])[["gene_HUGO_id", "ircc_id",	"gene_direction"]].drop_duplicates().sort_values("ircc_id").set_index("ircc_id")
# encode gene_dir as binary features
in_df = in_df.gene_HUGO_id.astype(str) + "_" + in_df.gene_direction.astype(str)
# account for multiple CNV events for each sample
CNV_matrix = pd.get_dummies(in_df).reset_index().groupby("ircc_id").sum()

# load drug response data
ctx3w_cat = drug_response_data[[
    "ircc_id", target_col]].    set_index("ircc_id")
# encode target col
Y_class_dict = {'PD': 0, 'SD': 1, 'OR': 1}
ctx3w_cat[target_col] = ctx3w_cat[target_col].replace(Y_class_dict)


feature_col = CNV_matrix.columns
features_in = pd.merge(ctx3w_cat, CNV_matrix,
                       right_index=True, left_index=True)
# replace na w/t 0
features_in = features_in.fillna(0)
# drop instances w/t missing target
features_in = features_in[~features_in[target_col].isna()
                          ].    drop_duplicates()
features_in.to_csv(snakemake.output.preproc_CNV,
                   sep='\t')
