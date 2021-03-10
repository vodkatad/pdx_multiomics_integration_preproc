#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Compute CNV stats for intogen genes that map to PDx segmented CN data (from CNVkit)

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


# scikit-learn
from sklearn.model_selection import train_test_split
# scalers
from sklearn.preprocessing import StandardScaler
# processing
from sklearn.pipeline import Pipeline
# models
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, Lasso
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# feature selection
from sklearn.feature_selection import SelectFromModel
# benchmark
from sklearn.metrics import roc_curve, multilabel_confusion_matrix, auc, matthews_corrcoef, roc_auc_score, accuracy_score
# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import pickle


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


# load sample id conversion table, drug response data
drug_response_data = pd.read_csv("tables/DrugResponse_LMXfirslevel_trainTest.csv", sep="\t")
        

# parse PDx segmented CNV data
f = "data/CNA_annotation/our_cn_genes2_CNVkitREDO_25012021.tsv"
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

# load shared set of genes for intogen and targeted sequencing
common_geneset = pd.read_table("tables/targeted_intogen_common_geneset.tsv", 
header=None, sep="\t")
common_geneset = common_geneset[0].tolist()

# merge with genealogy using Sanger ids 
PDx_CNV_data = pd.merge(PDx_CNV_data,
                       drug_response_data[["sanger_id", "sample_level", 
                       "ircc_id", "is_LMX", "ircc_id_short",
                       "passage",'lineage']].drop_duplicates(),
                       left_on = "sample",
                       right_on = "sanger_id")

# reshape into a gene x model CNV matrix
in_df = PDx_CNV_data[["gene_HUGO_id", "ircc_id", "log2R"]]
# merge multiple CNV values for same gene;model (TODO: check these)
in_df = PDx_CNV_data.groupby(["gene_HUGO_id", "ircc_id"]).agg({"log2R" : "mean"}).reset_index()
CNV_matrix = in_df.set_index(["ircc_id", "gene_HUGO_id"]).unstack()
CNV_matrix.columns = CNV_matrix.columns.get_level_values("gene_HUGO_id").tolist()
CNV_matrix.index = CNV_matrix.index.get_level_values("ircc_id").tolist()
CNV_matrix.to_csv("tables/irccID_CNlog2R_CNVkitREDO.tsv", sep="\t")

# load drug response data 
ctx3w_cat = drug_response_data[["ircc_id", "Cetuximab_Standard_3wks_cat"]].    set_index("ircc_id")
features_in = pd.merge(ctx3w_cat, CNV_matrix, right_index=True, left_index=True)
features_in.shape
features_in.head()


# In[3]:


len([g for g in CNV_matrix.columns if g in common_geneset])


# In[4]:


def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between Energy Star Score
    #y = x['score']
    #x = x.drop(columns = ['score'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    print(f"dropped {drops}")
    x = x.drop(columns = drops)
         
    return x


# In[5]:


genes_tokeep = [g for g in features_in.columns if g in common_geneset]
features_clean = features_in
target_col = "Cetuximab_Standard_3wks_cat"
features_col = np.array([c for c in features_clean.columns if c != target_col])
# remove features with 0 variance
features_clean = features_clean.loc[(features_clean.var(axis=1) == 0).index]
# remove colinear features 
features_clean = remove_collinear_features(features_clean[features_col], .75)
# add back genes in common geneset
genes_toadd = [c for c in genes_tokeep if c not in features_clean.columns]
features_clean = pd.concat([features_clean, features_in[genes_toadd]], axis=1)
features_col = features_clean.columns
# replace na w/t 0
features_clean = features_clean.fillna(0)
features_clean[target_col] = features_in[target_col]
# drop instances w/t missing target 
features_clean = features_clean[~features_clean[target_col].isna()].    drop_duplicates()
features_clean.shape


# In[6]:


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


# In[7]:


missing_values_table(features_clean)


# In[8]:


TT_df = drug_response_data[drug_response_data.ircc_id.isin(features_clean.index)][
    [ "ircc_id", "is_test"]]
                            
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()


# train-test split
X_train = features_clean.loc[train_models, features_col]
y_train  = features_clean.loc[train_models, target_col]
X_test = features_clean.loc[test_models, features_col]
y_test = features_clean.loc[test_models, target_col]


# standardise features
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train.values),
                        columns=features_col,
                        index=train_models)
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test.values),
                        columns=features_col,
                        index=test_models)
X_train.to_csv("tables/PDx_CNV_FeatSelect_Xtrain.tsv", sep="\t")
X_test.to_csv("tables/PDx_CNV_FeatSelect_X_test.tsv", sep="\t")
y_train.to_csv("tables/PDx_CNV_FeatSelect_Ytrain.tsv", sep="\t")
y_test.to_csv("tables/PDx_CNV_FeatSelect_Ytest.tsv", sep="\t")

X_train = X_train.values
Y_train = y_train.values
X_test = X_test.values
Y_test = y_test.values

# train linearSVM
svm = LinearSVC().fit(X_train, Y_train)


# In[9]:


X_test.shape


# In[10]:


y_classes = features_clean[target_col].unique().tolist()
Y_pred = svm.predict(X_test)

multi_cm = multilabel_confusion_matrix(Y_test, Y_pred, labels=y_classes)
tn, fp, fn, tp = [i for i in sum(multi_cm).ravel()]
accuracy = tp + tn / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
# harmonic mean of precion and recall
F1 = 2*(precision * recall) / (precision + recall)
model_mcc = matthews_corrcoef(Y_test, Y_pred)
printout = f"{svm} \n Precision: {precision:.4f} |Recall: {recall:.4f}  |MCC: {model_mcc:.4f}  |F1: {F1:.4f} |Accu: {accuracy:.4f}"
print(printout)


# In[11]:


features_col.shape


# In[12]:


# get linear SVC feature coefficients 
coeff_plot_df = pd.DataFrame(svm.coef_.T,
                            columns=svm.classes_, 
                            index=features_col)
coeff_plot_df = coeff_plot_df.stack().reset_index()
coeff_plot_df.columns=["feature", "class", "coeff"]
coeff_plot_df = coeff_plot_df.sort_values("coeff")
# select top / bottom features
top = pd.concat([coeff_plot_df.head(10), coeff_plot_df.tail(10)]).feature.unique()
plot_df = coeff_plot_df[coeff_plot_df.feature.isin(top)]

fig,ax = plt.subplots(figsize=(10,16))
ax = sb.barplot(x="coeff",
                y="feature", 
                hue="class",
                palette="Set2",
                data=plot_df)


# # Conclusions and Next Steps
# Summarize findings here

# In[ ]:




