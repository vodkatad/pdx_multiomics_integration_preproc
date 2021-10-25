#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Imports
# Import libraries and write settings here.

# In[48]:


# Data manipulation
import pandas as pd
import numpy as np
from scipy import stats

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30

# autoML
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from ConfigSpace.configuration_space import Configuration
import autosklearn.classification
import PipelineProfiler


# scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# models
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier
# xgboost
from xgboost import XGBClassifier

# processing
from sklearn.preprocessing import label_binarize, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import ColumnSelector
from sklearn import model_selection

# dimensionality reduction, clustering
import math
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, MeanShift, DBSCAN
from sklearn.neighbors import kneighbors_graph
import umap
# feature selection
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2

# benchmark
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, multilabel_confusion_matrix, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
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

# In[49]:


# load pre-computed meth cluster labels
# TODO re compute on train set only to prevent info leaks
f = "data/methylation/k5_samples-clusters_division.tsv"
k5_clusters = pd.read_csv(f, sep="\t", header=0)
# convert index to CRC id short
k5_clusters.index = [c[:-3] for c in k5_clusters.index.tolist()]
# encode cluster labels as binary features
k5_clusters = pd.get_dummies(k5_clusters.cluster)
k5_clusters.head()


# In[50]:


# load sample id conversion table, drug response data
drug_response_data = pd.read_csv("tables/DrugResponse_LMXfirslevel_trainTest.csv", sep="\t")
        
features_clean_df = pd.merge(drug_response_data[[
                            "Cetuximab_Standard_3wks_cat", "ircc_id_short", "ircc_id", "is_test"]],
                            k5_clusters,
                            left_on="ircc_id_short",
                            right_index=True)

# encode target
Y_class_dict={'PD':0,'SD':1, 'OR':1}
features_clean_df['Cetuximab_Standard_3wks_cat'] =  features_clean_df['Cetuximab_Standard_3wks_cat'].replace(Y_class_dict)      
               
train_models = features_clean_df[features_clean_df.is_test == False].ircc_id.unique()
test_models = features_clean_df[features_clean_df.is_test == True].ircc_id.unique()
features_clean_df = features_clean_df.drop(["is_test", "ircc_id_short"], axis=1).set_index("ircc_id")

features_clean_df.head()


# In[51]:


input_matrix = features_clean_df
input_matrix.index = input_matrix.index.values
target_col = "Cetuximab_Standard_3wks_cat"
features_col = np.array([c for c in input_matrix.columns if c != target_col])
# save processed features
features_clean_df[features_col].to_csv('tables/preprocessed_features/methK5Clusters.tsv',
                                          sep='\t')

# train-test split
X_train = input_matrix.loc[train_models, features_col].values
y_train  = input_matrix.loc[train_models, target_col].values
X_test = input_matrix.loc[test_models, features_col].values
y_test = input_matrix.loc[test_models, target_col].values

# scale features
X_train = MinMaxScaler().fit_transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)



# In[23]:


X_train.shape
X_test.shape


# In[24]:


# basic classifier accuracy test
lr = LogisticRegression().fit(X_train, y_train)
svm = LinearSVC().fit(X_train, y_train)
knc = KNeighborsClassifier().fit(X_train, y_train)
rfc = RandomForestClassifier().fit(X_train, y_train)
xgc = XGBClassifier().fit(X_train, y_train)
[(model, accuracy_score(y_test, model.predict(X_test))) for model in [lr, svm, knc, rfc, xgc]]


# In[31]:


# pipeline to train a classifier on meth data alone
pipe_steps = [
    ("VarianceFilter", VarianceThreshold(threshold=0)), # drop features with 0 variance
    ('KNNlassifier', KNeighborsClassifier().fit(X_train, y_train)),
]

pipeMeth = Pipeline(pipe_steps)


# In[35]:


hyperparameter_grid = {
          'KNNlassifier__n_neighbors' : list(range(1,30)),
          'KNNlassifier__p' : [1, 2, 3, 4, 5],
          #'RFClassifier__max_features' : np.linspace(.01, .8, 5, endpoint=True),
          #'RFClassifier__min_samples_split' :  np.linspace(.01, .5, 5, endpoint=True),
          }

# Set up the random search with 4-fold stratified cross validation
skf = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
grid = GridSearchCV(estimator=pipeMeth, 
                    param_grid=hyperparameter_grid, 
                    scoring="accuracy",
                    n_jobs=-1,
                    cv=skf,
                    refit=True,
                    verbose=2)


# In[36]:


grid.fit(X_train, y_train)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy on train: %.2f' % grid.best_score_)


# assess best classifier performance on test set
grid_test_score = grid.best_estimator_.score(X_test, y_test)
y_pred = grid.best_estimator_.predict(X_test)
print(f'Accuracy on test set: {grid_test_score:.3f}')
# print classification report on test set
print(classification_report(y_test, y_pred, target_names=['PD', 'SD-OR']))


# In[37]:


y_test_predict_proba = grid.predict_proba(X_test)
roc_auc_score(y_test, y_test_predict_proba[:, 1])


# # Conclusions and Next Steps
# Summarize findings here

# In[ ]:




