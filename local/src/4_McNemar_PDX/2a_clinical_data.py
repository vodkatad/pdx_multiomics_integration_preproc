#!/usr/bin/env python
# coding: utf-8

# # Introduction
# preprocess PDX models clinical data
# this 'omic' requires a lot more hand-processing
# since the ata are particularly spaarse and heterogeneous

# ### Imports
# Import libraries and write settings here.
# Data manipulation
import pandas as pd
import numpy as np

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30

# load sample id conversion table, drug response data
f = snakemake.input.response
drug_response_data = pd.read_csv(f, sep="\t")

# load PDX models clinical data
# this includes info on the original patient from which the
# the graft originates
f = snakemake.input.clin_data
clin_df = pd.read_excel(f, skiprows=1).dropna(
    how='all', axis=1).set_index("CRC Code")
# drop cols with cetuximab reponse (aka our target variable)
todrop = [c for c in clin_df.columns if "RESPONSE" in c and "CETUX" in c]
clin_df = clin_df.drop(todrop, axis=1)
# drop any date column, keep age at collectiopn
todrop = [c for c in clin_df.columns if "Date" in c] + \
    ["Age at resection", "Age at first diagnosis"]
clin_df = clin_df.drop(todrop, axis=1)

f = snakemake.input.site_annot
location_df = pd.read_csv(f, header=0, sep='\t')
location_df = location_df[['Case', 'site']].set_index('Case')
clin_df = pd.merge(clin_df, location_df, left_index=True, right_index=True)

# this clinical data is particularly sparse, so we drop features
# with too many missing values
# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


missing_df = missing_values_table(clin_df)
missing_df = missing_df[missing_df["% of Total Values"] < 40]
tokeep = missing_df.index.tolist()  # drop cols with more than 30% NaNs

# drop another few handpicked cols of uncertain significance / quality
todrop = [
    "N",
    "T",
    "N° of other metastatic resections before collected metastasis",
    "M",
    "SEDE M",
    "Site of primary",
    "Site of primary DICOT"
]
clin_df = clin_df[tokeep].drop(todrop, axis=1)

# ### clean up NaNs,NOs, encode categorical features
# keep only numerical(-like) levels for cancer stage feature
stage_dict = {
    '0': 0,
    "I": 1,
    "II": 2,
    "NAN": 5,  # everything else is a 5
    "IIIB": 3,
    "IIA": 2,
    "IIIC": 3,
    "IIB": 2,
    "IVB": 4,
    "IIIA": 3,
    "IIC": 2,
    "I-III": 3,  # highest stage in range
    "IVA": 4,
    "III": 3,
    "IV": 4,

}
clin_df["Stage at first diagnosis"] = clin_df["Stage at first diagnosis"].    apply(
    lambda x: str(x).upper()).replace(stage_dict)

# if both "high" and "low" are present in the description, assign "high"
# everyother level is a "5"
clin_df["Stage at first diagnosis"] = clin_df["Stage at first diagnosis"].    replace(to_replace=r'.*ALT.*', value=6, regex=True).        replace(
    to_replace=r'.*BASS.*', value=7, regex=True).            apply(lambda x: x if (isinstance(x, int)) else 5)
clin_df["Stage at first diagnosis"].unique()

# convert  Lymph node density (LND)
# (i.e. ratio of positive LN/total LN) to numerical score


def calc_LR(s):
    s = s.replace('O', '0')
    if s == 'NaN' or "SU" not in s:
        return np.nan
    elif 'SU0' in s:
        return 0
    # if there are multiple vals for sample
    elif ";" in s:
        pos_arr = []
        tot_arr = []
        for sub_s in s.split(";"):
            pos, tot = [float(s) for s in sub_s.split("SU")]
            pos_arr.append(pos)
            tot_arr.append(tot)
        # return mean score
        return np.mean(pos) / np.mean(tot)
    # return positive ln / tot ln as score
    pos, tot = [float(s) for s in s.split("SU")]
    return pos / tot


clin_df["lnd ratio"] = clin_df["lnd ratio"].astype(str).apply(calc_LR)
clin_df[["Max diameter (mm)", "N° of Metastases"]] = clin_df[["Max diameter (mm)", "N° of Metastases"]].\
    replace({'NO': 0, "multiple": 2, "NA": np.nan}).astype(float)
# clean up previous treatment cols
treat_cols = [c for c in clin_df.columns if 'treat' in c.lower()] +\
    ["Extra-hepatic metastases (Y, N, NA)",
     "ANY RADIO BEFORE COLLECTION (excluding neoadjuvant relative to collected resection; Y/N)"]
clin_df[treat_cols] = clin_df[treat_cols].\
    replace(to_replace=r' ', value='', regex=True).\
    replace(to_replace=r'.{10,}', value=np.nan, regex=True).\
    replace(to_replace=r'^Y\(.*$', value='Y', regex=True).\
    replace({"N": 0, "Y": 1, "NA": np.nan, '1o2': 2, "y": 1, "n": 0})
clin_df["Sex"] = clin_df["Sex"].replace({"F": 1, "M": 0, "M ": 0})
# encode sample anatomical location col
dummy_location_df = pd.get_dummies(clin_df["site"])
clin_df = pd.merge(clin_df.drop('site', axis=1),
                   dummy_location_df, left_index=True, right_index=True)

# Merge with drug response
feature_col = clin_df.columns
target_col = snakemake.params.target_col
features_in = pd.merge(drug_response_data[["ircc_id_short",
                                           "is_test",
                                           target_col,
                                           "ircc_id"]],
                       clin_df,
                       left_on="ircc_id_short",
                       right_index=True).\
    set_index("ircc_id").drop('ircc_id_short', axis=1)

# drop duplicated instances (ircc_id) from index
features_in = features_in[~features_in.index.duplicated(keep='first')]

# replace strings
features_in = features_in.replace('NO', 0).replace('NA', np.nan).\
    replace({"N": 0, "Y": 1, "NA": np.nan, '1o2': 2, "y": 1, "n": 0})
features_in[feature_col].to_csv(snakemake.output.preproc_clin,
                                sep='\t')
