#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Imports
# Import libraries and write settings here.

# In[60]:


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
import vaex

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

# In[61]:


# read methylation M file in chunks, conver to to hd5
#in_filename="data/methylation/m_values_Umberto.tsv"
#master_filename = 'data/methylation/m_values_Umberto.hd5'
#numerical_cols = pd.read_csv("data/methylation/columns.tsv", header=None)[0][1:].tolist()
#chunksize = 10000
#chunk_list = []
#i = 0
#for chunk in pd.read_csv(in_filename, 
#                sep="\t", 
#                iterator=True, 
#                chunksize=chunksize,
#                names=["probe"] + numerical_cols,
#                header=None, 
#                skiprows=1):
#        # covert to np float
#        chunk[numerical_cols] = chunk[numerical_cols].apply(pd.to_numeric, 
#                                                            errors='coerce', 
#                                                            downcast='float')
#        chunk["probe"] = chunk["probe"].astype(str)
#        vaex_df = vaex.from_pandas(chunk, copy_index=False)
#        out_filename = f"data/methylation/m_values/m_values_{i}.hd5"
#        vaex_df.export_hdf5(path=out_filename)
#        i+=1
#        chunk_list.append(out_filename)
#        del chunk
#master_df = vaex.open_many(chunk_list)
#master_df.export_hdf5(path=master_filename)


# In[62]:


master_filename = 'data/methylation/m_values_Umberto.hd5'
features_in = vaex.open(master_filename)
nrows, ncols = features_in.shape
print(f"Master dataset has {nrows} rows and {ncols} columns")
features_in.describe()


# In[63]:


# read methylation bionomial test pvalue 
f = "data/methylation/beta_DT-pvalue_Xeno.tsv"
bDTpval = pd.read_csv(f, sep="\t").reset_index()
bDTpval.columns=["probe", "beta_DT-pvalue"]
bDTpval = bDTpval.set_index("probe")
# read methylation M variance
f =  "data/methylation/m_sdvalue.tsv"
Msd = pd.read_csv(f, sep="\t").reset_index()
Msd.columns=["probe", "M_sd"]
Msd = Msd.set_index("probe")
probe_stats = pd.concat([Msd, bDTpval], axis=1)

# filter probes w/t binomial FDR < .05,
# 10% most variable probes
M_sd_thrs = probe_stats.describe(percentiles=[.95, .9, .75]).loc["75%", "M_sd"]
FDR = .05
probe_stats[(probe_stats["beta_DT-pvalue"]<FDR) &             (probe_stats["M_sd"]>M_sd_thrs)].shape


# In[64]:


# load sample id conversion table, drug response data
drug_response_data = pd.read_csv("tables/DrugResponse_LMXfirslevel_trainTest.csv", sep="\t")
        
# filter on probes, models
samples_tokeep = drug_response_data.ircc_id.apply(lambda x:x.replace("TUM", " ").split()[0]).unique()
samples_tokeep = [c for c in samples_tokeep if c in list(features_in.columns.keys())[1:]]
probes_tokeep = pd.Series(probe_stats[(probe_stats["beta_DT-pvalue"]<FDR) &                             (probe_stats["M_sd"]>M_sd_thrs)].index.tolist())
features_clean = features_in[features_in["probe"].isin(probes_tokeep)][samples_tokeep + ["probe"]]

# convert to pandas, reshape, add target
features_clean_df = features_clean.to_pandas_df()
features_clean_df = features_clean_df.set_index("probe").T
features_clean_df.columns = features_clean_df.columns.tolist()
features_clean_df["ircc_id_short"] = [x[0:7] for x in features_clean_df.index]
features_clean_df = pd.merge(drug_response_data[[
                            "Cetuximab_Standard_3wks_cat", "ircc_id_short", "ircc_id", "is_test"]],
                            features_clean_df,
                            on="ircc_id_short")
                            
train_models = features_clean_df[features_clean_df.is_test == False].ircc_id.unique()
test_models = features_clean_df[features_clean_df.is_test == True].ircc_id.unique()
features_clean_df = features_clean_df.drop(["is_test", "ircc_id_short"], axis=1).set_index("ircc_id")
features_clean_df.head()


# In[65]:


features_clean_df["Cetuximab_Standard_3wks_cat"].value_counts()


# In[66]:


test_models.shape


# In[67]:


input_matrix = features_clean_df
input_matrix.index = input_matrix.index.values
target_col = "Cetuximab_Standard_3wks_cat"
features_col = np.array([c for c in input_matrix.columns if c != target_col])

# train-test split
X_train = input_matrix.loc[train_models, features_col].values
y_train  = input_matrix.loc[train_models, target_col].values
X_test = input_matrix.loc[test_models, features_col].values
y_test = input_matrix.loc[test_models, target_col].values

# standardise features
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# In[68]:


X_test.shape


# In[69]:


# the smaller C the fewer features selected
lsvc = LinearSVC(C=0.1, 
penalty="l1", 
dual=False).fit(X_train, y_train)
selector = SelectFromModel(lsvc, prefit=True)
lsvc_selected = features_col[selector.get_support()]
lsvc_selected.shape
lsvc_selected


# In[70]:


# 2d PCA for gene expression (vsd), Cetuximab response class
# using only llsvc selected features
N = 4
pca = PCA(n_components=N)
X_new = input_matrix[lsvc_selected].values
principalComponents = pca.fit_transform(X_new)
PC_df = pd.DataFrame(data=principalComponents,
                     columns=['PC_' + str(n+1) for n in range(N)])
# add taget class col
PC_df[target_col] = y


fig, axes = plt.subplots(1, 2, figsize=(16, 8))
ax1, ax2 = axes
ax1 = sb.scatterplot(data=PC_df,
                     x="PC_1",
                     y="PC_2",
                     palette="Set2",
                     hue="Cetuximab_Standard_3wks_cat",
                     style="Cetuximab_Standard_3wks_cat",
                     ax=ax1)
ax2 = sb.scatterplot(data=PC_df,
                     x="PC_3",
                     y="PC_4",
                     palette="Set2",
                     hue="Cetuximab_Standard_3wks_cat",
                     style="Cetuximab_Standard_3wks_cat",
                     ax=ax2)
# explained variance annot
PC_1_var, PC_2_var, PC_3_var, PC_4_var = [
    e*100 for e in pca.explained_variance_ratio_]
xl = ax1.set_xlabel(f'PC1: {PC_1_var:.3f}% expl var', fontsize=12)
yl = ax1.set_ylabel(f'PC2: {PC_2_var:.3f}% expl var', fontsize=12)
xl = ax2.set_xlabel(f'PC3: {PC_3_var:.3f}% expl var', fontsize=12)
yl = ax2.set_ylabel(f'PC4: {PC_4_var:.3f}% expl var', fontsize=12)

st = fig.suptitle("lsvc feature selection")


# In[71]:


pipe_steps = [
    ("lSVCselector", SelectFromModel(
        LinearSVC(penalty="l1", dual=False))),  # feature selection
    ("pca", PCA()),
    ("SVCclassifier", SVC()),
]
hyperparameter_grid = {
    "lSVCselector__estimator__C": [.5, 0.1, 0.5, 0.05, 0.01],
    "pca__n_components": [2],
    "SVCclassifier__kernel": ["linear"],
    "SVCclassifier__C": [1, 0.1, 0.5, 0.01, 0.05, 0.01, 0.001, 0.05, 2, 5, 10, 30],
    "SVCclassifier__gamma": ["auto", 5, 2, 1, 0.1, 0.5, 0.01, 0.05]
}
pipeline = Pipeline(pipe_steps)

# Set up the random search with 4-fold cross validation
grid_cv = GridSearchCV(estimator=pipeline,
                               param_grid=hyperparameter_grid,
                               cv=4, #n_iter=100,
                               scoring="accuracy",
                               n_jobs=-1, refit=True,
                               return_train_score=True)
                               #random_state=42)

grid_cv.fit(X_train, y_train)


# In[72]:


CVresult_df = pd.DataFrame(grid_cv.cv_results_)
CVresult_df.sort_values("rank_test_score")[
    ["rank_test_score","mean_train_score", "mean_test_score"]].head()


# In[73]:


grid_cv_test_score = grid_cv.score(X_test, y_test)
grid_cv_test_score


# # Results
# Show graphs and stats here

# In[74]:


y_classes = input_matrix[target_col].unique().tolist()
Y_pred = grid_cv.predict(X_test)
# print (Y_pred)
multi_cm = multilabel_confusion_matrix(y_test, Y_pred, labels=y_classes)
tn, fp, fn, tp = [i for i in sum(multi_cm).ravel()]
accuracy = tp + tn / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
# harmonic mean of precion and recall
F1 = 2*(precision * recall) / (precision + recall)
model_mcc = matthews_corrcoef(y_test, Y_pred)
printout = f"{grid_cv.best_estimator_} \n Precision: {precision:.4f} |Recall: {recall:.4f}  |MCC: {model_mcc:.4f}  |F1: {F1:.4f} |Accu: {accuracy:.4f}"
print(printout)


# In[75]:


# plot the 2D decision boundary
# from: https://github.com/suvoooo/Machine_Learning/blob/master/SVM_Decision_Boundary/Decision_Boundary_SVM.ipynb

# rerun feature selection, pca
X_test_df = pd.DataFrame(X_test,
                         columns=features_col)
selector = grid_cv.best_estimator_[0]
lsvc_selected = features_col[selector.get_support()]
X_test_selected = X_test_df[lsvc_selected].values
pca2 = PCA(n_components=2)
X_test_selected_reduced = pca2.fit_transform(X_test_selected)

classify = grid_cv.best_estimator_[2]

def plot_contours(ax, clf, xx, yy, **params):
    Z = classify.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([cdict2[z] for z in Z])
    print('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


X0, X1 = X_test_selected_reduced[:, 0], X_test_selected_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('white')


Y_tar_list = y_test.tolist()
labels1 = pd.Series(Y_tar_list)

full_labels = {"OR": "Objective Response (<-50%)",
               "PD": "Progressive Disease (>35%)",
               "SD": "Stable Disease"}


arr = [pd.Series(a) for a in [labels1, X0, X1]]
plot_df = pd.concat(arr, axis=1)
plot_df.columns = ["true_class", "PC_1", "PC_2"]
hue_order = ["OR", "PD", "SD"]
cdict2 = dict(zip(hue_order, [1, 2, 3]))

# draw contours showing model predictions
cs = plot_contours(ax, classify, xx, yy,
                   colors=["limegreen", "peru", "w", "deepskyblue", "w"],
                   alpha=0.5, levels=3, label="Prediction")

# scatter PC1, PC2 coords for each test instance
ax = sb.scatterplot(data=plot_df,
                    x="PC_1",
                    y="PC_2",
                    hue_order=hue_order,
                    hue="true_class",
                    palette=["limegreen", "peru", "deepskyblue"],
                    alpha=1,
                    s=60,
                    zorder=20,
                    ax=ax)

# highlight support vector instances
sv = ax.scatter(classify.support_vectors_[:, 0],
                classify.support_vectors_[:, 1],
                s=60, facecolors='none',
                edgecolors='k', label='Support Vectors',
               zorder=100)

# make proxy artists for contours colours
proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
         for pc in cs.collections if pc.get_facecolor()[0][0] != 1]

# custom legend
h, l = ax.get_legend_handles_labels()
l = [full_labels.get(e, e) for e in l]
h = h + [h[0]] + proxy
l = l + ["prediction"] + hue_order # add subtitle for contours
plt.legend(h, l, fontsize=15, bbox_to_anchor=(1, 1.05))

xl = plt.xlabel("PC_1", fontsize=14)
yl = plt.ylabel("PC_2", fontsize=14)

# annotate w/t pipeline params
n_genes = lsvc_selected.shape[0]
classifier_params = grid_cv.best_estimator_[2].get_params()
kernel = classifier_params["kernel"]
gamma = classifier_params["gamma"]
C = classifier_params["C"]
accu = grid_cv_test_score
st = f"SVC n_genes={n_genes}, k={kernel}; γ={gamma}; C={C}; accuracy={accu:.3f} on PDx meth M"
st = fig.suptitle(st, y=.95, fontsize=18)

fig.tight_layout
fig.savefig('figures/PDx_DrugResponse_MethM_MultiFeatSelect_2dPCA_SVC_descisionBoundary.pdf',
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "PDx_Meth_MultiFeatSelect.ipynb"})


# In[76]:


model_filename = "models/MethM_Centuximab32_multiFeatSelect_lSVC.pkl"
with open(model_filename,'wb') as f:
    pickle.dump(grid_cv.best_estimator_,f)


# In[81]:


# load the model from disk
model_filename = "models/MethM_Centuximab32_multiFeatSelect_lSVC.pkl"
loaded_model = pickle.load(open(model_filename, 'rb'))
svm = loaded_model[2]
selector = loaded_model[0]
# get selected features
lsvc_selected = features_col[selector.get_support()]

# slice selected features, save train test
X_train = input_matrix.loc[train_models, lsvc_selected]
y_train  = input_matrix.loc[train_models, target_col]
X_test = input_matrix.loc[test_models, lsvc_selected]
y_test = input_matrix.loc[test_models, target_col]

# standardise features
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train.values),
                        columns=lsvc_selected,
                        index=train_models)
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test.values),
                        columns=lsvc_selected,
                        index=test_models)
X_train.to_csv("tables/PDx_Meth_MultiFeatSelect_Xtrain.tsv", sep="\t")
X_test.to_csv("tables/PDx_Meth_MultiFeatSelect_X_test.tsv", sep="\t")
y_train.to_csv("tables/PDx_Meth_MultiFeatSelect_Ytrain.tsv", sep="\t")
y_test.to_csv("tables/PDx_Meth_MultiFeatSelect_Ytest.tsv", sep="\t")


# In[77]:


# get the feature selector (linear SVC) coeff
coeff_plot_df = pd.DataFrame(selector.estimator_.coef_.T,
                            columns=svm.classes_, 
                            index=features_col)
# keep only supported features 
coeff_plot_df["support"] = selector.get_support()
coeff_plot_df = coeff_plot_df[coeff_plot_df["support"] == True][svm.classes_]
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




