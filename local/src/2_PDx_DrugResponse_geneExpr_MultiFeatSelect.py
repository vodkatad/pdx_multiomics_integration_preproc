#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Imports
# Import libraries and write settings here.

# In[1]:


# Data manipulation
import seaborn as sb
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
import numpy as np
from scipy import stats

# statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30

# Display all cell outputs
InteractiveShell.ast_node_interactivity = 'all'

ipython = get_ipython()

# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

# Visualizations
# Set default font size
plt.rcParams['font.size'] = 24
# Set default font size
sb.set(font_scale=.8)
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

# load expression data from RNAseq
# these are variance stabilized (vsd)
f = "data/RNAseq/release_2/selected_matrix.tsv"
rnaseq_matrix = pd.read_csv(f, sep="\t", header=0, index_col=0)
rnaseq_matrix = rnaseq_matrix.T.reset_index().    rename(columns={'index': 'ircc_id'})
rnaseq_matrix["ircc_id_short"] = rnaseq_matrix.ircc_id.apply(lambda x: x[0:7])
rnaseq_matrix = rnaseq_matrix.drop("ircc_id", axis=1)

# merge expression and Centuximab 3w response
merge_matrix = pd.merge(rnaseq_matrix,
                        drug_response_data[[
                            "Cetuximab_Standard_3wks_cat", "ircc_id_short", "ircc_id"]],
                        on="ircc_id_short")

# drop instances w/t missing target value
merge_matrix = merge_matrix[~merge_matrix["Cetuximab_Standard_3wks_cat"].isna()]
merge_matrix = merge_matrix.drop("ircc_id_short", axis=1).    set_index("ircc_id").drop_duplicates()
merge_matrix.to_csv("tables/RNAseqVSD_irccLong_Cetuximab3wCat.tsv", sep="\t")
merge_matrix.shape
merge_matrix.head()


# In[3]:


merge_matrix["Cetuximab_Standard_3wks_cat"].value_counts()


# In[4]:


input_matrix = merge_matrix
input_matrix.index = input_matrix.index.values
target_col = "Cetuximab_Standard_3wks_cat"
features_col = np.array([c for c in input_matrix.columns if c != target_col])

TT_df = drug_response_data[drug_response_data.ircc_id.isin(input_matrix.index)][
    [ "ircc_id", "is_test"]]
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()

# train-test split
X_train = input_matrix.loc[train_models, features_col]
y_train  = input_matrix.loc[train_models, target_col]
X_test = input_matrix.loc[test_models, features_col]
y_test = input_matrix.loc[test_models, target_col]

# standardise features
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train.values),
                        columns=features_col,
                        index=X_train.index)
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test.values),
                        columns=features_col,
                        index=X_test.index)

X_train = X_train.values
Y_train = y_train.values
X_test = X_test.values
Y_test = y_test.values


# Expression and methylation data are too high-dimensional to integrate.
# We need to perform feature selection for these two OMICs. I'm using SelectFromModel to select features based on importance.
# I'm using LinearSVC as a model b/c it seems to have the best performance on unselected expression data (i.e. it's able to find the most important features). Linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero. When the goal is to reduce the dimensionality of the data to use with another classifier, they can be used along with SelectFromModel to select the non-zero coefficients. In particular, sparse estimators useful for this purpose are the Lasso for regression, and of LogisticRegression and LinearSVC for classification.
# 
# Other options:
#     - tree-based feature selection
#     - sequential feature selection
# 

# In[5]:


# the smaller C the fewer features selected
lsvc = LinearSVC(C=0.1, 
penalty="l1", 
dual=False).fit(X_train, y_train)
selector = SelectFromModel(lsvc, prefit=True)
lsvc_selected = features_col[selector.get_support()]
lsvc_selected.shape
lsvc_selected


# # Results
# Show graphs and stats here

# 

# In[ ]:





# In[9]:


# 2d PCA for gene expression (vsd), Cetuximab response class
# using only llsvc selected features
N = 4
pca = PCA(n_components=N)
X_new = input_matrix[lsvc_selected].values
principalComponents = pca.fit_transform(X_new)
PC_df = pd.DataFrame(data=principalComponents,
                     columns=['PC_' + str(n+1) for n in range(N)])
# add taget class col
PC_df[target_col] = input_matrix[target_col].values


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


# In[10]:


feat_var = np.var(principalComponents, axis=0)
feat_var_rat = feat_var / (np.sum(feat_var))
feat_var_rat


# The first 2 PC are most relevant for separating samples by Centuximab 3w repsonse. Together they explain ~63% of the variance.

# In[20]:


pipe_steps = [
    ("lSVCselector", SelectFromModel(
        LinearSVC(penalty="l1", dual=False))),  # feature selection
    ("pca", PCA()),
    ("SVCclassifier", SVC()),
]
hyperparameter_grid = {
    "lSVCselector__estimator__C": [0.1, 0.5, 0.05],
    "pca__n_components": [2],
    "SVCclassifier__kernel": ["linear"],
    "SVCclassifier__C": [1, 0.1, 0.5, 0.01, 0.05, 0.01, 0.05, 10, 30, 40],
    "SVCclassifier__gamma": ["auto", 1, 0.1, 0.5, 0.01, 0.05]
}
pipeline = Pipeline(pipe_steps)

# Set up grid search with 4-fold cross validation
grid_cv = GridSearchCV(estimator=pipeline,
                               param_grid=hyperparameter_grid,
                               cv=4, #n_iter=100,
                               scoring="accuracy",
                               n_jobs=-1, refit=True,
                               return_train_score=True)
                               #random_state=42)

grid_cv.fit(X_train, y_train)


# In[21]:


CVresult_df = pd.DataFrame(grid_cv.cv_results_)
CVresult_df.sort_values("rank_test_score")[
    ["rank_test_score","mean_train_score", "mean_test_score"]].head()


# In[22]:


grid_cv_test_score = grid_cv.score(X_test, y_test)
grid_cv_test_score


# In[23]:


print(grid_cv.best_params_)


# In[24]:


grid_cv.best_estimator_[2].get_params()


# In[25]:


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


# In[26]:


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
st = f"SVC n_genes={n_genes}, k={kernel}; γ={gamma}; C={C}; accuracy={accu:.3f} on PDx vsd expression"
st = fig.suptitle(st, y=.95, fontsize=18)

fig.tight_layout
fig.savefig('figures/PDx_DrugResponse_geneExpr_MultiFeatSelect_2dPCA_SVC_descisionBoundary.pdf',
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "PDx_DrugResponse_geneExpr_MultiFeatSelect.ipynb"})


# In[27]:


model_filename = "models/geneExpr_vsd_Centuximab32_multiFeatSelect_lSVC.pkl"
with open(model_filename,'wb') as f:
    pickle.dump(grid_cv.best_estimator_,f)


# # TODO try  SequentialFeatureSelector as in 
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py

# In[7]:


# load the model from disk
model_filename = "models/geneExpr_vsd_Centuximab32_multiFeatSelect_lSVC.pkl"
loaded_model = pickle.load(open(model_filename, 'rb'))
svm = loaded_model[2]
selector = loaded_model[0]

# get selected feature
input_matrix = merge_matrix
input_matrix.index = input_matrix.index.values
target_col = "Cetuximab_Standard_3wks_cat"

TT_df = drug_response_data[drug_response_data.ircc_id.isin(input_matrix.index)][
    [ "ircc_id", "is_test"]]
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()

features_col = np.array([c for c in input_matrix.columns if c != target_col])
lsvc_selected = features_col[selector.get_support()]

# slice selected features, save train test
X_train = input_matrix.loc[train_models, lsvc_selected]
y_train  = input_matrix.loc[train_models, target_col]
X_test = input_matrix.loc[test_models, lsvc_selected]
y_test = input_matrix.loc[test_models, target_col]

# standardise features
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train.values),
                        columns=lsvc_selected,
                        index=X_train.index)
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test.values),
                        columns=lsvc_selected,
                        index=X_test.index)
X_train.to_csv("tables/PDx_Expr_MultiFeatSelect_Xtrain.tsv", sep="\t")
X_test.to_csv("tables/PDx_Expr_MultiFeatSelect_X_test.tsv", sep="\t")
y_train.to_csv("tables/PDx_Expr_MultiFeatSelect_Ytrain.tsv", sep="\t")
y_test.to_csv("tables/PDx_Expr_MultiFeatSelect_Ytest.tsv", sep="\t")


# In[10]:


StandardScaler().fit_transform(X_test.values)


# In[31]:


selector.get_support().shape


# In[32]:


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
