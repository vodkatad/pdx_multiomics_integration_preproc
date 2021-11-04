# Data manipulation
import pandas as pd
import numpy as np
import vaex # read hd5 file

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30

# load all methylation probe data
f = snakemake.input.meth
features_in = vaex.open(f)

# load sample id conversion table, drug response data
f = snakemake.input.response
drug_response_data = pd.read_csv(f, sep="\t")
# filter on response models IDs
samples_tokeep = drug_response_data.ircc_id.apply(
	lambda x:x.replace("TUM", " ").split()[0]).unique()
samples_tokeep = [
	c for c in samples_tokeep if c in list(
		features_in.columns.keys())[1:]]
features_clean = features_in[
		samples_tokeep + ["probe"]]

# convert to pandas, reshape, add target var
features_clean_df = features_clean.to_pandas_df()
features_clean_df = features_clean_df.set_index("probe").T
features_clean_df.columns = features_clean_df.columns.tolist()
features_clean_df["ircc_id_short"] = [
	x[0:7] for x in features_clean_df.index]
features_clean_df = pd.merge(drug_response_data[[
                            "Cetuximab_Standard_3wks_cat", 
			    "ircc_id_short", "ircc_id", 
			    "is_test"]],
                            features_clean_df,
                            on="ircc_id_short")
features_clean_df = features_clean_df.drop(
	["is_test", "ircc_id_short"], axis=1).set_index("ircc_id")
features_clean_df.to_csv(snakemake.output.raw_meth, sep='\t')