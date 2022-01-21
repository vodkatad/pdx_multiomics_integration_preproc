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

# read methylation bionomial test pvalue 
f = snakemake.input.bDTpval 
bDTpval = pd.read_csv(f, sep="\t").reset_index()
bDTpval.columns=["probe", "beta_DT-pvalue"]
bDTpval = bDTpval.set_index("probe")
# read methylation M variance
f = snakemake.input.Msd
Msd = pd.read_csv(f, sep="\t").reset_index()
Msd.columns=["probe", "M_sd"]
Msd = Msd.set_index("probe")
probe_stats = pd.concat([Msd, bDTpval], axis=1)
# read probe to gene map
f = snakemake.input.probe_gene_map
collapsed_ALLprobes = pd.read_csv(f, sep="\t", 
	header=0, index_col=0).reset_index()

# keep most variable probes
sd_pctl =  snakemake.params.sd_pctl
M_sd_thrs = probe_stats.describe(percentiles=[float(sd_pctl[:-1])/100]).\
    loc[sd_pctl, "M_sd"]
FDR = .05
probes_tokeep = pd.Series(probe_stats[(
    probe_stats["beta_DT-pvalue"] < FDR) # null hypothesis: dist is unimodal
    # & (probe_stats["M_sd"] > M_sd_thrs)
         ].index.tolist())

# keep only probes that are the 'best probe' for a given gene
probe_gene = collapsed_ALLprobes[collapsed_ALLprobes.probe.isin(probes_tokeep)]
probe_gene["feature"] = collapsed_ALLprobes["index"] + '_' + collapsed_ALLprobes.probe
probes_tokeep = probe_gene.probe.unique()
# build dict 
probe2feature = dict(zip(probe_gene.probe, probe_gene.feature))

# load sample id conversion table, drug response data
f = snakemake.input.response
drug_response_data = pd.read_csv(f, sep="\t")
target_col = snakemake.params.target_col
# filter on response models IDs
samples_tokeep = drug_response_data.ircc_id.apply(
	lambda x:x.replace("TUM", " ").split()[0]).unique()
samples_tokeep = [
	c for c in samples_tokeep if c in list(
		features_in.columns.keys())[1:]]
features_clean = features_in[features_in["probe"].\
	isin(probes_tokeep)][samples_tokeep + ["probe"]]

# map probes to gene names
features_clean['probe'] = features_clean['probe'].apply(lambda x:probe2feature[x])

# convert to pandas, reshape, add target var
features_clean_df = features_clean.to_pandas_df()
features_clean_df = features_clean_df.set_index("probe").T
features_clean_df.columns = features_clean_df.columns.tolist()
features_clean_df["ircc_id_short"] = [
	x[0:7] for x in features_clean_df.index]
features_clean_df = pd.merge(drug_response_data[[
                            target_col, 
			    "ircc_id_short", "ircc_id", 
			    "is_test"]],
                            features_clean_df,
                            on="ircc_id_short")
features_clean_df = features_clean_df.drop(
	[target_col,"is_test", "ircc_id_short"], axis=1).set_index("ircc_id")
features_clean_df.to_csv(snakemake.output.raw_meth, sep='\t')