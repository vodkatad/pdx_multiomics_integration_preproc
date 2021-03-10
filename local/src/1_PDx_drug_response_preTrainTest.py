#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Imports
# Import libraries and write settings here.

# In[1]:


# Data manipulation
import pandas as pd
import numpy as np
from scipy import stats

# statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from IPython import get_ipython
ipython = get_ipython()

# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

# Visualizations
import matplotlib.pyplot as plt
# Set default font size
plt.rcParams['font.size'] = 24
import seaborn as sb
# Set default font size
sb.set(font_scale = 1.2)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)


# Interactive Visualizations
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import iplot, init_notebook_mode
# init_notebook_mode(connected=True)

# import cufflinks as cf
# cf.go_offline(connected=True)
# icf.set_config_file(theme='white')


# # Analysis/Modeling
# Do work here

# In[2]:


# load sample id conversion table, genealogy data
f = "data/mapping_sanger_ID_genealogy_long.tsv"
id_data = pd.read_csv(f, sep="\t", names=["sanger_id", "ircc_id", "sample_level"])
# eval if sample is liver metastasis in PDx
id_data["is_LMX"] = id_data.ircc_id.str.contains("LMX")
# short ids consist of the first 7 chars of full ids
id_data["ircc_id_short"] =  id_data.ircc_id.apply(lambda x:x[0:7])
id_data["passage"] = id_data.ircc_id.apply(lambda x:int(x[12:14]))
id_data["lineage"] = id_data.ircc_id.apply(lambda x:x[10:12])
# filter out non-LMX models, firstlevel only
id_data = id_data[(id_data.is_LMX==True) & 
                (id_data.sample_level=="firstlevel")]

# load drug response data for PDx models
f = "data/drug_response/Treatments_Eugy_Ele_fix0cetuxi_201005.tsv"
drug_response_data = pd.read_csv(f, "\t", header=0).rename(columns={'CRC CODE':
                                                                    'ircc_id_short',
                                                                    "Cetuximab Standard 3wks" :
                                                                    "Cetuximab_Standard_3wks"})
# merge response data w/t gnenealogy data
drug_response_data = pd.merge(id_data,
                        drug_response_data[[
                            "Cetuximab_Standard_3wks", "ircc_id_short"]],
                        on="ircc_id_short")
# drop mpde;s w/t missing target
drug_response_data =  drug_response_data[~drug_response_data.Cetuximab_Standard_3wks.isna()] 

# encode target variable
# Objective Response, Progressive Disease & Stable Disease
m = drug_response_data.Cetuximab_Standard_3wks.min()
M = drug_response_data.Cetuximab_Standard_3wks.max()
drug_response_data["Cetuximab_Standard_3wks_cat"] = pd.cut(
    drug_response_data["Cetuximab_Standard_3wks"],
    bins=[m-1, -50, 35, M+1],
    labels=["OR", "SD", "PD"])

# train/test split 
n_test = int(drug_response_data.ircc_id.nunique() * .25)
test_models = drug_response_data.ircc_id.sample(n=n_test, random_state=13)
drug_response_data["is_test"] = drug_response_data["ircc_id"].isin(test_models)
drug_response_data.to_csv("tables/DrugResponse_LMXfirslevel_trainTest.csv", sep="\t", index=None)
drug_response_data  


# In[3]:


drug_response_data["is_test"].value_counts()


# # Results
# Show graphs and stats here

# In[7]:


drug_response_data = pd.read_csv("tables/DrugResponse_LMXfirslevel_trainTest.csv", sep="\t")
test = drug_response_data[drug_response_data["is_test"] == True]
train = drug_response_data[drug_response_data["is_test"] == False]
test.set_index("ircc_id")["Cetuximab_Standard_3wks_cat"].    to_csv("tables/Cetuximab_3w_cat_test.tsv", sep="\t")
train.set_index("ircc_id")["Cetuximab_Standard_3wks_cat"].    to_csv("tables/Cetuximab_3w_cat_train.tsv", sep="\t")


# # Conclusions and Next Steps
# Summarize findings here

# In[ ]:




