#!/usr/bin/env python
# coding: utf-8

# Data manipulation
import pandas as pd

# load sample id conversion table, drug response data
drug_response_data = pd.read_csv(snakemake.input.response,
                                 sep="\t")
# load expression data from RNAseq
# these are variance stabilized (vsd)
rnaseq_matrix = pd.read_csv(snakemake.input.expr,
                            sep="\t", header=0, index_col=0)
rnaseq_matrix = rnaseq_matrix.T.reset_index(
).    rename(columns={'index': 'ircc_id'})
rnaseq_matrix["ircc_id_short"] = rnaseq_matrix.ircc_id.apply(lambda x: x[0:7])
rnaseq_matrix = rnaseq_matrix.drop("ircc_id", axis=1)

target_col = snakemake.params.target_col
# merge expression and Centuximab 3w response
merge_matrix = pd.merge(rnaseq_matrix,
                        drug_response_data[[
                            target_col, "ircc_id_short", "ircc_id"]],
                        on="ircc_id_short")

# drop instances w/t missing target value
merge_matrix = merge_matrix[~merge_matrix[target_col].isna()]
merge_matrix = merge_matrix.drop(
    "ircc_id_short", axis=1).    set_index("ircc_id").drop_duplicates()
merge_matrix.to_csv(snakemake.output.preproc_expr, sep="\t")
