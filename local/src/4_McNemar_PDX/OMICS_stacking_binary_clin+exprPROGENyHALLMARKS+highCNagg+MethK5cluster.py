#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Train, optimise stacked predictor of Cetuximab sensitivity

# ### Imports
# Import libraries and write settings here.

# Data manipulation
import pandas as pd
import numpy as np
import warnings
# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30


# scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# models
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from mlxtend.classifier import LogisticRegression as extLogisticRegression

# processing
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import ColumnSelector
from sklearn import model_selection

# feature selection
from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold, chi2

# feature agglomeration
from sklearn.cluster import FeatureAgglomeration

# benchmark
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, multilabel_confusion_matrix, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import pickle


## Analysis/Modeling
## load all 'omics preprocessed datasets

# K5 clusters encoded meth probes
f = snakemake.input.meth
Meth = pd.read_csv(f, sep="\t", header=0, index_col=0)
Meth = Meth[Meth.columns.drop(list(Meth.filter(regex='Cetuximab')))]

# encoded expr data w/t progeny pathway scores + msdb hallmarks ssGSEA scores
# processed through a colinearity + chi2 filter (drop the worst of each colinear pair of features)
f = snakemake.input.expr
Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr = Expr[Expr.columns.drop(list(Expr.filter(regex='Cetuximab')))]
Expr.columns = [c + "_expr" for c in Expr.columns]

# feature agglomeration CNV, input includes highGain features (> than 1 copy gained)
f = snakemake.input.cnv
CNV = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV = CNV[CNV.columns.drop(list(CNV.filter(regex='Cetuximab')))]
CNV.columns = [c + "_cnv" for c in CNV.columns]

# custom mut feature cross w/t top 20 features by chi2
f = snakemake.input.mut
Mut = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut = Mut[Mut.columns.drop(list(Mut.filter(regex='Cetuximab')))]
Mut.columns = [c + "_mut" for c in Mut.columns]

# add clinical data ()custom encoding, filtering)
f = snakemake.input.clin
Clin = pd.read_csv(f, sep="\t", header=0, index_col=0)
Clin = Clin[Clin.columns.drop(list(Clin.filter(regex='Cetuximab')))]
Clin.columns = [c + "_clin" for c in Clin.columns]

# load drug response data
f = snakemake.input.response
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode target var (binary responder/non-responder)
Y_class_dict={'PD':0,'OR+SD':1}
Y[target_col] = Y[target_col].replace(Y_class_dict)

# merge all feature blocks + response together
df1 = pd.merge(Mut, CNV, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth, Expr, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
all_df = pd.merge(all_df, Clin, right_index=True, left_index=True, how="outer")
feature_col = all_df.columns.tolist()
all_df =  all_df.select_dtypes([np.number])
all_df = pd.merge(all_df, Y[target_col], right_index=True, left_index=True, how="right")

# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]
# fill sparse features with median imputation
all_df[feature_col] = all_df[feature_col].\
    astype(float).apply(lambda col:col.fillna(col.median()))

# train-test split
train_models = Y[Y.is_test == False].index.unique()
test_models = Y[Y.is_test == True].index.unique()
X_train = all_df.loc[train_models, feature_col]
y_train  = all_df.loc[train_models, target_col]
X_test = all_df.loc[test_models, feature_col]
y_test = all_df.loc[test_models, target_col]

#scale features separately
scaler = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train.values),
          columns=X_train.columns, index=X_train.index)              
X_test = pd.DataFrame(scaler.transform(X_test.values),
          columns=X_test.columns, index=X_test.index)    

# log train, test shape, dataset balance
X_train.shape
X_test.shape

# get indeces for feature subsets, one per OMIC
Meth_indeces = list(range(0, Meth.shape[1]))
pos = len(Meth_indeces)
Expr_indeces = list(range(Meth_indeces[-1]+1, pos + Expr.shape[1]))
pos += len(Expr_indeces)
Mut_indeces = list(range(Expr_indeces[-1]+1, pos + Mut.shape[1]))
pos += len(Mut_indeces)
CNV_indeces = list(range(Mut_indeces[-1]+1, pos + CNV.shape[1]))
pos += len(CNV_indeces)
Clin_indeces = list(range(CNV_indeces[-1]+1, pos + Clin.shape[1]))

# log n of features for each block

# pipeline to train a classifier on meth data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Meth_indeces)),
    ("VarianceFilter", VarianceThreshold(threshold=0)), # drop features with 0 variance
    ('KNNlassifier', KNeighborsClassifier().fit(X_train, y_train)),
]

pipeMeth = Pipeline(pipe_steps)


# In[63]:


# pipeline to train a classifier on expression data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Expr_indeces)),
    ("chi2filterFscore", SelectKBest(chi2)), 
    ('RFClassifier', RandomForestClassifier(criterion='gini', class_weight='balanced')),
]

pipeExpr = Pipeline(pipe_steps)


# In[64]:


# pipeline to train a classifier on mutation data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Mut_indeces)),
    ("chi2filterFscore", SelectKBest(chi2)), # univariate filter on chi2 stat
    ('RFClassifier', RandomForestClassifier(criterion='gini', class_weight='balanced')),
]

pipeMut = Pipeline(pipe_steps)


# In[65]:


# pipeline to train a classifier on CNV data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=CNV_indeces)),
     # remove samples which have the same val in 85% or more samples
    ("VarianceFilter", VarianceThreshold(threshold=(.75 * (1 - .75)))),
    ("WardAgg", FeatureAgglomeration()), # Ward feature agglomeration by mean
    ("chi2filterFscore", SelectKBest(chi2)), 
    ('RFClassifier', RandomForestClassifier(criterion='gini', class_weight='balanced')),
]

pipeCNV = Pipeline(pipe_steps)


# In[66]:


# pipeline to train a classifier on clinical/patient data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Clin_indeces)),
    ("chi2filterFscore", SelectKBest(chi2)), 
    ('RFClassifier', RandomForestClassifier(criterion='gini', class_weight='balanced')),
]

pipeClin = Pipeline(pipe_steps)

# build the meta classifier
sclf = StackingCVClassifier(classifiers=[pipeMeth, pipeExpr, pipeMut, pipeCNV, pipeClin], 
                          cv=3, random_state=13, verbose=1,
                          #use_probas=True, #average_probas=False,
                          #use_features_in_secondary=True,
                          meta_classifier=LogisticRegression(penalty='l2', class_weight='balanced'))

hyperparameter_grid = {
          # Meth params 
          'pipeline-1__KNNlassifier__n_neighbors' : [13],
          'pipeline-1__KNNlassifier__p' : [1],
          #'pipeline-1__': [],
          # Expr params 
          'pipeline-2__chi2filterFscore__k': [25],
          'pipeline-2__RFClassifier__max_features' : [.2],
          'pipeline-2__RFClassifier__min_samples_split' : [.4],
          'pipeline-2__RFClassifier__n_estimators' : [15],
          'pipeline-2__RFClassifier__max_depth' : [11],
          #'pipeline-2__': [],
          # Mut params
          'pipeline-3__chi2filterFscore__k': [50],
          'pipeline-3__RFClassifier__max_features' : [.2],
          'pipeline-3__RFClassifier__min_samples_split' : [.01],
          'pipeline-3__RFClassifier__n_estimators' : [15],
          'pipeline-3__RFClassifier__max_depth' : [11],
          #'pipeline-3__' : [],
          # CNV params
          'pipeline-4__WardAgg__n_clusters' : [40],
          'pipeline-4__chi2filterFscore__k': [12],
          'pipeline-4__RFClassifier__max_features' : [.4],
          'pipeline-4__RFClassifier__min_samples_split' : [.4],
          'pipeline-4__RFClassifier__n_estimators' : [10],
          'pipeline-4__RFClassifier__max_depth' : [3],
          # clin params
          'pipeline-5__chi2filterFscore__k': [10],
          'pipeline-5__chi2filterFscore__k': [25],
          'pipeline-5__RFClassifier__max_depth' : [4],
          # meta classifier params
          'meta_classifier__C':  np.linspace(.01, .9, 10, endpoint=True)
          }

# Set up the random search with 4-fold stratified cross validation
skf = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
grid = GridSearchCV(estimator=sclf, 
                    param_grid=hyperparameter_grid, 
                    n_jobs=-1,
                    cv=skf,
                    refit=True,
                    verbose=2)
grid.fit(X_train, y_train)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))


print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)

# pickle best model from gridCV
model_filename = "models/stacked_Omics_binary_MultiCVclassifier_clin+exprPROGENyHALLMARKS+highCNagg+MethK5cluster.pkl" 

with open(model_filename,'wb') as f:
    pickle.dump(grid.best_estimator_,f)

# load the model from file
classifier = pickle.load(open(model_filename, 'rb'))
# assess best classifier performance on test set
grid_test_score = classifier.score(X_test, y_test)
y_pred = classifier.predict(X_test)
print(f'Accuracy on test set: {grid_test_score:.3f}')
# print classification report on test set
print(classification_report(y_test, y_pred, target_names=['PD', 'SD-OR']))

#confusion_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=['PD', 'SD-OR'],
                                 cmap=plt.cm.Blues)


# In[72]:


# Learn to predict e/a class against e/a other
#OVRclassifier = OneVsRestClassifier(classifier)
# returns the marginal probability that the given sample has the label in question
y_test_predict_proba = classifier.predict_proba(X_test)
roc_auc_score(y_test, y_test_predict_proba[:, 1])


# # Conclusions and Next Steps
# Summarize findings here

# In[ ]:




