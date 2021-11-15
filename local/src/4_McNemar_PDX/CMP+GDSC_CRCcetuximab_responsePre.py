#!/usr/bin/env python
# coding: utf-8

# preprocess 

# ### Imports
# Import libraries and write settings here.

# Data manipulation
import pandas as pd
import numpy as np
import math

from scipy.stats import zscore


# load cell line metadata from cellmodelpassport
f = snakemake.input.CMP_meta
CMP_meta_data = pd.read_csv(f, header=0)
CMP_meta_data.head()

# get CRC line names, IDs
CMP_CRC_lines = CMP_meta_data[CMP_meta_data.cancer_type == "Colorectal Carcinoma"].model_name.unique()
CMP_CRC_sangerID = CMP_meta_data[CMP_meta_data.cancer_type == "Colorectal Carcinoma"].model_id.unique()

# add GDSC drug response data
GDSC_response_data = pd.concat([pd.read_csv("data/GDSC/GDSC1_fitted_dose_response_25Feb20.csv"),
                                pd.read_csv("data/GDSC/GDSC2_fitted_dose_response_25Feb20.csv")]).drop_duplicates()
GDSC_response_data = GDSC_response_data.rename(
    columns={"DRUG_NAME": "DRUG_NAME_response"})

# obtain IC50
GDSC_response_data["IC50"] = GDSC_response_data["LN_IC50"].astype(float). apply(lambda x: math.exp(x))
# check if IC50<max_c
GDSC_response_data["IC50<MAX_CONC"] = GDSC_response_data.IC50 <     GDSC_response_data.MAX_CONC
# compute the IC50/MAX_C ratio
GDSC_response_data["IC50_ratio"] = GDSC_response_data.IC50 /     GDSC_response_data.MAX_CONC
GDSC_response_data.COSMIC_ID = GDSC_response_data.COSMIC_ID.astype(
    int).astype(str)
# filter for CMP CRC lines screened with Cetuximab
GDSC_cetux_lines = GDSC_response_data[GDSC_response_data.DRUG_NAME_response == 'Cetuximab'].CELL_LINE_NAME.unique()
GDSC_cetux_sangerID = GDSC_response_data[GDSC_response_data.DRUG_NAME_response == 'Cetuximab'].SANGER_MODEL_ID.unique()
CRC_cetux_sangerID = set(CMP_CRC_sangerID).intersection(GDSC_cetux_sangerID)
GDSC_CRCcetuxi_response_data = GDSC_response_data[(GDSC_response_data.SANGER_MODEL_ID.isin(CRC_cetux_sangerID) &
		(GDSC_response_data.DRUG_NAME_response == 'Cetuximab'))]

# generate a binary target variable by
# splitting models by IC50 median (placeholder)
IC50_median = GDSC_CRCcetuxi_response_data.IC50.median()
GDSC_CRCcetuxi_response_data['IC50_byMedian_cat'] = (GDSC_CRCcetuxi_response_data.IC50 < IC50_median).replace({True : 1, False : 0})
GDSC_CRCcetuxi_response_data['IC50_byMedian_cat'].value_counts()
# calc IC50 Zscore
GDSC_CRCcetuxi_response_data['IC50_Zscore'] = zscore(GDSC_CRCcetuxi_response_data.IC50.values)
sb.displot(data=GDSC_CRCcetuxi_response_data,
	x='IC50_Zscore',
	 kind="kde")

# write to file
out_df = GDSC_CRCcetuxi_response_data[[
	'SANGER_MODEL_ID',
	'DATASET',
	'IC50',
	'IC50_ratio',
	'MAX_CONC',
	'IC50<MAX_CONC',
	'IC50_byMedian_cat',
	'IC50_Zscore'
]]
testSize = .3
out_df.to_csv('tables/DrugResponse_CMP+GDSC_CRCcetuximab_trainTest.csv', sep="\t", index=None)
