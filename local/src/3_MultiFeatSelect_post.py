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
sb.set(font_scale = .8)
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


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[3]:


f = "tables/PDx_Meth_MultiFeatSelect_Xtrain.tsv"
Meth_train = pd.read_csv(f, sep="\t", header=0, index_col=0)
f = "tables/PDx_Expr_MultiFeatSelect_Xtrain.tsv"
Expr_train = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr_train.columns = [c + "_expr" for c in Expr_train.columns]
f = "tables/PDx_CNV_FeatSelect_Xtrain.tsv"
CNV_train = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV_train.columns = [c + "_cnv" for c in CNV_train.columns]
f = "tables/PDx_driverMutVAF_FeatSelect_Xtrain.tsv"
Mut_train = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut_train.columns = [c + "_mut" for c in Mut_train.columns]
f = "tables/Cetuximab_3w_cat_train.tsv"
Y_train = pd.read_csv(f, sep="\t", index_col=0, header=0)

df1 = pd.merge(Mut_train, CNV_train, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth_train, Expr_train, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
all_df = pd.merge(all_df, Y_train, right_index=True, left_index=True, how="right")

target_col = "Cetuximab_Standard_3wks_cat"
feature_col = [c for c in all_df.columns if c != target_col]
all_df[feature_col] = all_df[feature_col].apply(lambda col:col.fillna(col.median()))
# drop duplicated index vals
all_df = all_df[~all_df.index.duplicated(keep='first')]

print(all_df.shape)
print(missing_values_table(all_df))

f = "tables/PDx_Meth_MultiFeatSelect_XtrainClean.tsv"
Meth_train_clean = all_df[Meth_train.columns]
Meth_train_clean.to_csv(f, sep="\t")

f = "tables/PDx_Expr_MultiFeatSelect_XtrainClean.tsv"
Expr_train_clean = all_df[Expr_train.columns]
Expr_train_clean.to_csv(f, sep="\t")

f = "tables/PDx_CNV_FeatSelect_XtrainClean.tsv"
CNV_train_clean = all_df[CNV_train.columns]
CNV_train_clean.to_csv(f, sep="\t")

f = "tables/PDx_driverMutVAF_FeatSelect_XtrainClean.tsv"
Mut_train_clean = all_df[Mut_train.columns]
Mut_train_clean.to_csv(f, sep="\t")

f = "tables/Cetuximab_3w_cat_trainClean.tsv"
Y_train_clean = all_df["Cetuximab_Standard_3wks_cat"]
Y_train_clean.to_csv(f, sep="\t")


# In[4]:


f = "tables/PDx_Meth_MultiFeatSelect_Xtest.tsv"
Meth_test = pd.read_csv(f, sep="\t", header=0, index_col=0)
f = "tables/PDx_Expr_MultiFeatSelect_Xtest.tsv"
Expr_test = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr_test.columns = [c + "_expr" for c in Expr_test.columns]
f = "tables/PDx_CNV_FeatSelect_Xtest.tsv"
CNV_test = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV_test.columns = [c + "_cnv" for c in CNV_test.columns]
f = "tables/PDx_driverMutVAF_FeatSelect_Xtest.tsv"
Mut_test = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut_test.columns = [c + "_mut" for c in Mut_test.columns]
f = "tables/Cetuximab_3w_cat_test.tsv"
Y_test = pd.read_csv(f, sep="\t", index_col=0, header=0)

df1 = pd.merge(Mut_test, CNV_test, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth_test, Expr_test, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
all_df = pd.merge(all_df, Y_test, right_index=True, left_index=True, how="right")

target_col = "Cetuximab_Standard_3wks_cat"
feature_col = [c for c in all_df.columns if c != target_col]
all_df[feature_col] = all_df[feature_col].apply(lambda col:col.fillna(col.median()))
# drop duplicated index vals
all_df = all_df[~all_df.index.duplicated(keep='first')]

print(all_df.shape)
print(missing_values_table(all_df))

f = "tables/PDx_Meth_MultiFeatSelect_XtestClean.tsv"
Meth_test_clean = all_df[Meth_test.columns]
Meth_test_clean.to_csv(f, sep="\t")

f = "tables/PDx_Expr_MultiFeatSelect_XtestClean.tsv"
Expr_test_clean = all_df[Expr_test.columns]
Expr_test_clean.to_csv(f, sep="\t")

f = "tables/PDx_CNV_FeatSelect_XtestClean.tsv"
CNV_test_clean = all_df[CNV_test.columns]
CNV_test_clean.to_csv(f, sep="\t")

f = "tables/PDx_driverMutVAF_FeatSelect_XtestClean.tsv"
Mut_test_clean = all_df[Mut_test.columns]
Mut_test_clean.to_csv(f, sep="\t")

f = "tables/Cetuximab_3w_cat_testClean.tsv"
Y_test_clean = all_df["Cetuximab_Standard_3wks_cat"]
Y_test_clean.to_csv(f, sep="\t")


# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here

# In[ ]:




