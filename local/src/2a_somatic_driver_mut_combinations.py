#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Assess impact of mutation combinations on Centuximab 3w

# ### Imports
# Import libraries and write settings here.

# In[11]:


# Data manipulation
import pandas as pd
import numpy as np
from scipy import stats

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
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# load sample id conversion table, drug response data
drug_response_data = pd.read_csv("tables/DrugResponse_LMXfirslevel_trainTest.csv", sep="\t")
        

# load driver annotation for PDx models
f = "data/Driver_Annotation/CodingVariants_All669PDX_samples_26Feb2020_annotated_drivers_shortVersionForPDXfinder_EK.txt"
driver_data = pd.read_csv(f, "\t", header=0).rename(columns={'Sample':
'sanger_id'})

# + driver annot data
drug_mut_df = pd.merge(drug_response_data, driver_data, 
                       on="sanger_id", how="left")

# transform df to have a gene x sample binary mutation matrix
# including all driver genes
features_pre = drug_mut_df[drug_mut_df["Final_Driver_Annotation"] == True][
    ["ircc_id",
     "Gene",
     "Cetuximab_Standard_3wks",
     "Cetuximab_Standard_3wks_cat"]
].drop_duplicates()

# 1-hot encode genes, vector sum on sample to
# account for multiple mut in same sample
features_in = pd.get_dummies(features_pre.Gene)
features_in["ircc_id"] = features_pre.ircc_id
features_in = features_in.groupby("ircc_id").sum()

# add cetuximab response as target
df1 = features_pre[["Cetuximab_Standard_3wks", "Cetuximab_Standard_3wks_cat", "ircc_id"]
                   ].drop_duplicates().set_index("ircc_id")
features_in = pd.concat([features_in, df1], axis=1).drop_duplicates()
features_in.shape
features_in.head()


# In[3]:


# how many KRAS + BRAF + NRAS + APC mutants?
features_in[["KRAS", "BRAF", "NRAS", "APC"]].value_counts()


# In[4]:


wt_df = features_in[(features_in.KRAS == 0) &                     (features_in.BRAF == 0) &                     (features_in.NRAS == 0)]["Cetuximab_Standard_3wks"]
arr = []
for g in features_pre.Gene.unique():
    # is the mean Cetuximab_3w response of mutants different from wt's?
    # Mann–Whitney U test
    # null -> the probability of X being greater than Y is equal to 
    # the probability of Y being greater than X.
    mut = features_in[(features_in[g] == 1)]["Cetuximab_Standard_3wks"]
    non_mut = wt_df
    s,p = stats.mannwhitneyu(mut, non_mut)
    mm = mut.median()
    wm = non_mut.median()
    arr.append([g, s, p, mm, wm])
MW_df = pd.DataFrame(arr, columns=["gene","MW_stat", "MWU_pval", "mut_med", "wt_med"])

# is the mean Cetuximab_3w response of mutants different from wt's?
plot_df = MW_df.sort_values("MWU_pval").head(20)

# createcustom  colormap for p.adjust
range_min = plot_df["MWU_pval"].min()
range_max = plot_df["MWU_pval"].max()
norm = mcolors.Normalize(range_min, range_max)
# clusterProfiler style red-violet-blue cmap
cmap = LinearSegmentedColormap.from_list(
        "RVB", [(1, 0, 0), "#8F00FF", (0, 0, 1)], N=20)
cmap = matplotlib.cm.ScalarMappable(
      norm = norm, 
      cmap = cmap)

fig,ax=plt.subplots(figsize=(6,8))
ax = sb.barplot(x="mut_med", y="gene", color="b", data=MW_df.head(20))
yl = ax.set_ylabel("")
xl = ax.set_xlabel("Cetuximab_Standard_3wks_med")

# color each bar according to cmap
for container in ax.containers:
        for i, child  in enumerate(container.get_children()):
            child.set_fc(cmap.to_rgba(plot_df["MWU_pval"].values[i]))
            
# add cbar axis, cbar
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(cmap,
             cax=cax, 
             orientation='vertical', 
             label='MWU pval')

st = fig.suptitle("Cetuximab 3wks median response by mutational status", fontsize=14)
fig.savefig("figures/SingleDriverMut_cetuxi3w_barplot.pdf", format="pdf",
            bbox_inches='tight', dpi=300)


# In[5]:


wt_df = features_in[(features_in.KRAS == 0) &                     (features_in.BRAF == 0) &                     (features_in.NRAS == 0)]["Cetuximab_Standard_3wks"]
def MWU_group(values):
    mut = values
    non_mut = wt_df
    s,p = stats.mannwhitneyu(mut, non_mut)
    mm = mut.median()
    wm = non_mut.median()
    return s,p,mm,wm
MWU_df = features_in.groupby(["KRAS", "BRAF", "NRAS"]
                    ).Cetuximab_Standard_3wks.agg(MWU_group).apply(pd.Series)
MWU_df.columns = ["U_stat", "MWU_pval", "Cetuximab_Standard_3wks_med", "triple_neg_med"]
MWU_df["name"] = ["triple_neg", "NRAS", "BRAF", "NRAS + BRAF", "KRAS", "KRAS + NRAS"]
MWU_df


# # Results
# Show graphs and stats here

# In[6]:


# is the mean Cetuximab_3w response of mutants different from wt's?
plot_df = MWU_df.sort_values("MWU_pval")
# createcustom  colormap for p.adjust
range_min = plot_df["MWU_pval"].min()
range_max = plot_df["MWU_pval"].max()
norm = mcolors.Normalize(range_min, range_max)
# clusterProfiler style red-violet-blue cmap
cmap = LinearSegmentedColormap.from_list(
        "RVB", [(1, 0, 0), "#8F00FF", (0, 0, 1)], N=20)
cmap = matplotlib.cm.ScalarMappable(
      norm = norm, 
      cmap = cmap)

fig,ax=plt.subplots(figsize=(6,8))
ax = sb.barplot(x="Cetuximab_Standard_3wks_med", y="name", color="b", data=plot_df)

yl = ax.set_ylabel("")
xl = ax.set_xlabel("Cetuximab_Standard_3wks_med")

# color each bar according to cmap
for container in ax.containers:
        for i, child  in enumerate(container.get_children()):
            child.set_fc(cmap.to_rgba(plot_df["MWU_pval"].values[i]))
            
# add cbar axis, cbar
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(cmap,
             cax=cax, 
             orientation='vertical', 
             label='MWU pval')

st = fig.suptitle("Cetuximab 3wks response by mutational status", fontsize=14)
fig.savefig("figures/KRAS+NRAS+BRAF_combo_cetuxi3w_barplot.pdf", format="pdf",
            bbox_inches='tight', dpi=300)


# In[7]:


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


# In[8]:


features_clean = features_in.drop("Cetuximab_Standard_3wks", axis=1)
target_col = "Cetuximab_Standard_3wks_cat"
features_col = np.array([c for c in features_clean.columns if c != target_col])
# remove colinear features 
features_clean = remove_collinear_features(features_clean[features_col], .6)
# remove features with 0 variance
features_clean = features_clean.loc[(features_clean.var(axis=1) == 0).index]
features_col = features_clean.columns
features_clean["Cetuximab_Standard_3wks_cat"] = features_in["Cetuximab_Standard_3wks_cat"]

TT_df = drug_response_data[drug_response_data.ircc_id.isin(features_clean.index)][
    [ "ircc_id", "is_test"]]
                            
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()

# train-test split
X_train = features_clean.loc[train_models, features_col]
y_train  = features_clean.loc[train_models, target_col]
X_test = features_clean.loc[test_models, features_col]
y_test = features_clean.loc[test_models, target_col]

# no need to standardise features, they're binary already 
X_train.to_csv("tables/PDx_driverMut_FeatSelect_Xtrain.tsv", sep="\t")
X_test.to_csv("tables/PDx_driverMut_FeatSelect_X_test.tsv", sep="\t")
y_train.to_csv("tables/PDx_driverMut_FeatSelect_Ytrain.tsv", sep="\t")
y_test.to_csv("tables/PDx_driverMut_FeatSelect_Ytest.tsv", sep="\t")

X_train = X_train.values
Y_train = y_train.values
X_test = X_test.values
Y_test = y_test.values

# train linearSVM
svm = LinearSVC().fit(X_train, Y_train)


# In[9]:


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


# In[10]:


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
