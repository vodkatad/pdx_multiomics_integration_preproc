#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Imports
# Import libraries and write settings here.

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Train, optimise stacked predictor of Cetuximab sensitivity
import pandas as pd
import numpy as np
import warnings
# optuna + visulization
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import plotly.graph_objects as go
import ipywidgets as ipyw

from catboost import CatBoostClassifier
import pickle
#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, multilabel_confusion_matrix, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile, VarianceThreshold, chi2, SelectFromModel
from sklearn import model_selection
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from mlxtend.classifier import LogisticRegression as extLogisticRegression
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import no_type_check_decorator

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30


# Visualizations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sb
# Set default font size
plt.rcParams['font.size'] = 24
# Set default font size
sb.set(font_scale=.8)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)


# # Analysis/Modeling
# Do work here

# In[2]:


datadir = '../../../dataset/6_tuning/'
suffix = 0
class input:
        mut = datadir + f"preproc_mut{suffix}.tsv"
        cnv = datadir + f"preproc_CNV.tsv"
        raw_expr = datadir + "expr_merge.tsv"
        expr = datadir + f"preproc_expr{suffix}.tsv"
        meth = datadir + f"preproc_meth{suffix}.tsv"
        clin = datadir + f"preproc_clin{suffix}.tsv"
        response = datadir + f"DrugResponse_LMXfirslevel_trainTest{suffix}.tsv"
class output:
	X_train = f"mutCross+clin+exprPROGENyHALLMARKS+highCNagg+MethK5cluster{suffix}_Xtrain.tsv"
	X_test = f"mutCross+clin+exprPROGENyHALLMARKS+highCNagg+MethK5cluster{suffix}_Xtest.tsv"
	Y_train = f"OmicsBinary{suffix}_Ytrain.tsv"
	Y_test = f"OmicsBinary{suffix}_Ytest.tsv"
	best_model = f"OmicsBinary_StackingCVClassifier_mutCross+clin+exprPROGENyHALLMARKS+highCNagg+MethK5cluster{suffix}.pkl"
class params:
        target_col = "Cetuximab_Standard_3wks_cat"


# In[30]:


## Analysis/Modeling
## load all 'omics preprocessed datasets
# K5 clusters encoded meth probes
f = input.meth
Meth = pd.read_csv(f, sep="\t", header=0, index_col=0)
Meth = Meth[Meth.columns.drop(list(Meth.filter(regex='Cetuximab')))]
# encoded expr data w/t progeny pathway scores + msdb hallmarks ssGSEA scores
# processed through a colinearity + chi2 filter (drop the worst of each colinear pair of features)
f = input.expr
Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr = Expr[Expr.columns.drop(list(Expr.filter(regex='Cetuximab')))]
Expr.columns = [c + "_expr" for c in Expr.columns]
# raw expression data (variance-stabilised RNAseq)
f = input.raw_expr
raw_Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
raw_Expr = raw_Expr[raw_Expr.columns.drop(list(raw_Expr.filter(regex='Cetuximab|ircc_id_short')))]
raw_Expr.columns = [c + "_rawExpr" for c in raw_Expr.columns]
# feature agglomeration CNV, input includes highGain features (> than 1 copy gained)
f = input.cnv
CNV = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV = CNV[CNV.columns.drop(list(CNV.filter(regex='Cetuximab')))]
CNV.columns = [c + "_cnv" for c in CNV.columns]
# custom mut feature cross w/t top 20 features by chi2
f = input.mut
Mut = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut = Mut[Mut.columns.drop(list(Mut.filter(regex='Cetuximab')))]
Mut.columns = [c + "_mut" for c in Mut.columns]
# add clinical data (custom encoding, filtering)
f = input.clin
Clin = pd.read_csv(f, sep="\t", header=0, index_col=0)
Clin = Clin[Clin.columns.drop(list(Clin.filter(regex='Cetuximab')))]
Clin.columns = [c + "_clin" for c in Clin.columns]
# load drug response data
f = input.response
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode target var (binary responder/non-responder)
target_col = params.target_col
Y_class_dict={'PD':0,'OR+SD':1}
Y[target_col] = Y[target_col].replace(Y_class_dict)

# merge all feature blocks + response together
df1 = pd.merge(Mut, CNV, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth, Expr, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
all_df = pd.merge(all_df, Clin, right_index=True, left_index=True, how="outer")

feature_col = all_df.columns.tolist()
all_df = pd.merge(all_df, Y[target_col], right_index=True, left_index=True, how="right")
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]
# fill sparse features with median imputation
all_df[feature_col] = all_df[feature_col].    astype(float).apply(lambda col:col.fillna(col.median()))
# force to numeric
all_df = all_df.select_dtypes([np.number])
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
logfile = 'stacked_input.log'
with open(logfile, "w") as log:
    log.write(f"There are {X_train.shape[0]} instances in the trainig set."+ '\n')
    log.write(f"There are {X_test.shape[0]} instances in the test set."+ '\n')
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()  
    log.write(f"There are {train_counts.loc[0]} 'PD' instances and         {train_counts.loc[1]} 'SD+OR' instances in the training set."+ '\n')
    log.write(f"There are {test_counts.loc[0]} 'PD' instances and         {test_counts.loc[1]} 'SD+OR' instances in the test set."+ '\n')
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
with open(logfile, "a") as log:
    log.write(f"There are {X_train.shape[1]} total features."+ '\n')
    log.write(f"There are {Meth.shape[1]} methylation features."+ '\n')
    log.write(f"There are {Expr.shape[1]} Hallmarks, PROGENy expression features."+ '\n')
    log.write(f"There are {Mut.shape[1]} mutation features."+ '\n')
    log.write(f"There are {CNV.shape[1]} copy number features."+ '\n')
    log.write(f"There are {Clin.shape[1]} clinical features."+ '\n') 


# In[5]:


def calc_variance(X, Y):
    return pd.DataFrame(X).var()


fitted_models = []
# objective function for optuna


def objective(trial):
    # parameters to optimize
    # trial.suggest_int("Meth_KNNlassifier__n_neighbors", 5, 20, step=5)
    Meth_KNNlassifier__n_neighbors = 12
    # trial.suggest_int("Expr__chi2filterFscore__k", 14, 24)
    Expr__chi2filterFscore__k = 15
    # trial.suggest_int("Mut__chi2filterFscore__k", 2, 8, step=2)
    Mut__chi2filterFscore__k = 5
    # trial.suggest_int("CNV__WardAgg__n_clusters", 55, 85, step=5)
    CNV__WardAgg__n_clusters = 75
    # trial.suggest_int("CNV__chi2filterFscore__k", 25, 45, step=5)
    CNV__chi2filterFscore__k = 25
    # trial.suggest_int("Clin__chi2filterFscore__k", 4, 12, step=2)
    Clin__chi2filterFscore__k = 10
    meta__l1R = trial.suggest_loguniform("meta__C", 0.01, 1)
    #meta__penalty = trial.suggest_categorical("meta__penalty", ['l1', 'l2'])
    #trial.suggest_categorical("meta__use_probas", [True, False])
    #meta__secondary = trial.suggest_categorical("meta__use_secondary", [True, False]) 
    meta__use_probas = True
    #Mut_rf_n_estimators = trial.suggest_int(
    #    "Mut_rf_n_estimators", 10, 500, log=True)
    #Mut_rf_max_depth = trial.suggest_int("Mut_rf_max_depth", 1, 32, log=True)
    #Expr_rf_n_estimators = trial.suggest_int(
    #    "Expr_rf_n_estimators", 10, 500, log=True)
    Expr_rf_max_depth = trial.suggest_int("Expr_rf_max_depth", 1, 32, log=True)
    #Clin_rf_n_estimators = trial.suggest_int(
    #    "Clin_rf_n_estimators", 10, 500, log=True)
    #Clin_rf_max_depth = trial.suggest_int("Clin_rf_max_depth", 1, 32, log=True)
    CNV_rf_n_estimators = trial.suggest_int(
        "CNV_rf_n_estimators", 10, 500, log=True)
    CNV_rf_max_depth = trial.suggest_int("CNV_rf_max_depth", 1, 32, log=True)
    #Meta__n_estimators = trial.suggest_categorical("Meta__n_estimators", [100, 1000, 10000])
    #Meta__learning_rate=trial.suggest_float("Meta__learning_rate", 0.01, 0.3)
    #Meta__max_depth=trial.suggest_int("Meta__max_depth", 3, 12)

    # build stacked model pipeline
    # pipeline to train a classifier on meth data alone
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=Meth_indeces)),
        ('KNNlassifier', KNeighborsClassifier(
            n_neighbors=Meth_KNNlassifier__n_neighbors)),
    ]
    pipeMeth = Pipeline(pipe_steps)

    # pipeline to train a classifier on Hallmarks, PROGENy scores (expression)
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=Expr_indeces)),
        ("chi2filterFscore", SelectKBest(chi2, k=Expr__chi2filterFscore__k)),
        ('RFClassifier', RandomForestClassifier(max_depth=Expr_rf_max_depth))]
    pipeExpr = Pipeline(pipe_steps)

    # pipeline to train a classifier on mutation data alone
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=Mut_indeces)),
        # univariate filter on chi2 stat
        ("chi2filterFscore", SelectKBest(chi2, k=Mut__chi2filterFscore__k)),
        ('RFClassifier', CatBoostClassifier())#RandomForestClassifier(
            #max_depth=Mut_rf_max_depth, n_estimators=Mut_rf_n_estimators)),
    ]
    pipeMut = Pipeline(pipe_steps)

    # pipeline to train a classifier on CNV data alone
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=CNV_indeces)),
        # remove samples which have the same val in 5% or more samples
        ("VarianceFilter", VarianceThreshold(threshold=(.75 * (1 - .75)))),
        # Ward feature agglomeration by mean
        ("WardAgg", FeatureAgglomeration(n_clusters=CNV__WardAgg__n_clusters)),
        ("chi2filterFscore", SelectKBest(chi2, CNV__chi2filterFscore__k)),
        ('RFClassifier', RandomForestClassifier(max_depth=CNV_rf_max_depth, n_estimators=CNV_rf_n_estimators)),
    ]
    pipeCNV = Pipeline(pipe_steps)

    # pipeline to train a classifier on clinical/patient data alone
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=Clin_indeces)),
        ("chi2filterFscore", SelectKBest(chi2, k=Clin__chi2filterFscore__k)),
        ('RFClassifier', RandomForestClassifier()),
    ]
    pipeClin = Pipeline(pipe_steps)

    # build the meta classifier
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=13)
    sclf = StackingCVClassifier(classifiers=[
        pipeMeth,
        pipeExpr,
        # pipeRawExpr,
        pipeMut,
        pipeCNV,
        pipeClin
    ],
        cv=skf,
        n_jobs=-1,
        shuffle=True,
        random_state=13,
        verbose=0,
        use_probas=True,
        # average_probas=False,
        #use_features_in_secondary=meta__secondary,
        meta_classifier=LogisticRegression(
            penalty='elasticnet', 
            solver='saga',
            l1_ratio=meta__l1R))

    # fit on train, test return ROC AUC
    sclf = sclf.fit(X_train, y_train)
    #train_auc = roc_auc_score(y_train, sclf.predict(X_train))
    train_auc = cross_val_score(
        sclf, X_train, y_train, scoring='roc_auc', n_jobs=-1, cv=2).mean()
    test_auc = roc_auc_score(y_test, sclf.predict(X_test))
    fitted_models.append([test_auc, train_auc, sclf])
    return train_auc


# In[6]:



mlflow_cb = MLflowCallback(
    tracking_uri='mlruns',
    metric_name='ROC AUC'
)


CS_study = optuna.create_study(
    study_name='Cetuximab_sensitivity_prediction',
    direction='maximize',
    pruner=optuna.pruners.HyperbandPruner(max_resource="auto")
)


CS_study.optimize(objective, n_trials=50, callbacks=[mlflow_cb], n_jobs=50)


# # Results
# Show graphs and stats here

# In[9]:


CS_study.best_value


# In[18]:


CS_study.best_params


# In[10]:


fig1 = optuna.visualization.plot_optimization_history(CS_study)
fw1 = go.FigureWidget(fig1)
fw1.update_layout(width=610, margin=dict(r=250))

fig2 = optuna.visualization.plot_param_importances(CS_study)
fw2 = go.FigureWidget(fig2)
fw2.update_layout(margin=dict(r=100))

fig_subplots=ipyw.HBox([fw2, fw1])
fig_subplots


# In[19]:


best_model_df = pd.DataFrame(fitted_models, 
	columns=['test_auc', 'train_auc', 'fitted_model_obj']).\
		sort_values('train_auc', ascending=False)
best_model = best_model_df.iloc[0].fitted_model_obj
best_model_df.head()


# In[24]:


sb.lineplot(data=best_model_df, x=best_model_df.index, y="test_auc")


# In[23]:


sb.lineplot(data=best_model_df, x=best_model_df.index, y="train_auc")


# In[12]:


classifier = best_model
# assess best classifier performance on test set
test_score = classifier.score(X_test, y_test)
y_pred = classifier.predict(X_test)
print(f'Accuracy on test set: {test_score:.3f}')


# In[13]:


# print classification report on test set
print(classification_report(y_test, y_pred, target_names=['PD', 'SD-OR']))


# In[14]:


##confusion_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(classifier, X_test, y_test,
    display_labels=['PD', 'SD-OR'],
    cmap=plt.cm.Blues)


# In[15]:


triple_neg_CM = confusion_matrix(y_test, X_test['KRAS_BRAF_NRAS_triple_neg_mut'])
FP = triple_neg_CM.sum(axis=0) - np.diag(triple_neg_CM)  
FN = triple_neg_CM.sum(axis=1) - np.diag(triple_neg_CM)
TP = np.diag(triple_neg_CM)
TN = triple_neg_CM.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
tripleNeg_TPR = TP/(TP+FN)
# Fall out or false positive rate
tripleNeg_FPR = FP/(FP+TN)


# In[16]:


# return the marginal probability that the given sample has the label in question
y_pred = classifier.predict(X_test)
y_test_predict_proba = classifier.predict_proba(X_test)
fp_rates, tp_rates, _ = roc_curve(y_test,y_test_predict_proba[:,1])
roc_auc = auc(fp_rates, tp_rates)

fig, ax = plt.subplots(figsize=(8,8))
plt.plot(fp_rates, tp_rates, color='green',
            lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')

#plot decision point:
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = [i for i in cm.ravel()]
plt.plot(fp/(fp+tn), 
	tp/(tp+fn), 
	marker='o',
	color='darkgreen', 
	markersize=8, 
	label='Stacked decision point')

# add triple negative baseline
#ax.axvline(tripleNeg_FPR[1], ls=':', c='k', 
#        label='KRAS-BRAF-NRAS tripleNeg median')
plt.plot(tripleNeg_FPR[1], 
	tripleNeg_TPR[1], 
	marker='o',
	c='violet', 
	markersize=8,
	zorder=10,
	label='KRAS-BRAF-NRAS tripleNeg decision point')
#ax.axhline(tripleNeg_TPR[1], ls=':', c='k', 
#        label='KRAS-BRAF-NRAS tripleNeg median')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size=13)
plt.ylabel('True Positive Rate', size=13)
plt.title(f'ROC Curve on PDX test set (N={X_test.shape[0]})', size=15)
plt.legend(loc="lower right", prop={'size': 10})
plt.subplots_adjust(wspace=.3)


plt.savefig("bestStackedCVClassifier_vs_TripleNeg_AUC.pdf", 
                format='pdf', dpi=720, bbox_inches='tight')


# In[17]:


# pickle best model from trials
model_filename = output.best_model
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)


# In[31]:


X_test.shape


# In[44]:


f = "CR_mutCross+clin+exprPROGENyHALLMARKS+highCNagg+MethK5cluster"
XtestFile = "../../../dataset/5_McNemar_PDX_30x/" + f + str(suffix) + "_Xtest.tsv"
X_testCR = pd.read_csv(XtestFile, sep="\t", header=0, index_col=0)
f = "CR_OmicsBinary"
YtestFile = "../../../dataset/5_McNemar_PDX_30x/" + f + str(suffix) + "_Ytest.tsv"
y_testCR = pd.read_csv(YtestFile, sep="\t", header=0,
				index_col=0)['cetuxi_recist']


# In[54]:


y_predCR = classifier.predict(X_testCR)
y_testCR_predict_proba = classifier.predict_proba(X_testCR)
fp_ratesCR, tp_ratesCR, _ = roc_curve(y_testCR,y_testCR_predict_proba[:,1])
roc_auc = auc(fp_ratesCR, tp_ratesCR)

triple_neg_CM = confusion_matrix(y_testCR, X_testCR['KRAS_BRAF_NRAS_triple_neg_mut'])
FP = triple_neg_CM.sum(axis=0) - np.diag(triple_neg_CM)  
FN = triple_neg_CM.sum(axis=1) - np.diag(triple_neg_CM)
TP = np.diag(triple_neg_CM)
TN = triple_neg_CM.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
tripleNeg_TPR = TP/(TP+FN)
# Fall out or false positive rate
tripleNeg_FPR = FP/(FP+TN)

fig, ax = plt.subplots(figsize=(8,8))
plt.plot(fp_ratesCR, tp_ratesCR, color='green',
            lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')

#plot decision point:
cm = confusion_matrix(y_testCR, y_predCR)
tn, fp, fn, tp = [i for i in cm.ravel()]
plt.plot(fp/(fp+tn), 
	tp/(tp+fn), 
	marker='o',
	color='darkgreen', 
	markersize=8, 
	label='Stacked decision point')

plt.plot(tripleNeg_FPR[1], 
	tripleNeg_TPR[1], 
	marker='o',
	c='violet', 
	markersize=8,
	zorder=10,
	label='KRAS-BRAF-NRAS tripleNeg decision point')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size=13)
plt.ylabel('True Positive Rate', size=13)
plt.title(f'ROC Curve on PDX test set (N={X_test.shape[0]})', size=15)
plt.legend(loc="lower right", prop={'size': 10})
plt.subplots_adjust(wspace=.3)


# # Conclusions and Next Steps
# Summarize findings here

# In[ ]:




