#!/usr/bin/env python
# coding: utf-8
import os
# Data manipulation
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import graphviz
import seaborn as sb
import pandas as pd
import numpy as np
import warnings

# scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# models
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier, StackingCVClassifier

# processing
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import ColumnSelector
from sklearn import model_selection

# feature selection
from sklearn.feature_selection import f_classif, SelectKBest, SelectFromModel, VarianceThreshold, chi2

# benchmark
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# model persistence
import pickle

# model explaination
# shap for force diagram
import shap
# LIME for explaining predictions
import lime
import lime.lime_tabular
# LIME provides local model interpretability.
# LIME modifies a single data sample by tweaking
# the feature values and observes the resulting impact on the output.
# It tries to answer why was this prediction made / which variables
# caused this prediction

# Visualizations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# Set default font size
plt.rcParams['font.size'] = 24
# Set default font size
sb.set(font_scale=.8)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)


f = snakemake.input.meth
Meth = pd.read_csv(f, sep="\t", header=0, index_col=0)
Meth = Meth[Meth.columns.drop(list(Meth.filter(regex='Cetuximab')))]

f = snakemake.input.expr
Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr = Expr[Expr.columns.drop(list(Expr.filter(regex='Cetuximab')))]
Expr.columns = [c + "_expr" for c in Expr.columns]

f = snakemake.input.cnv
CNV = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV = CNV[CNV.columns.drop(list(CNV.filter(regex='Cetuximab')))]
CNV.columns = [c + "_cnv" for c in CNV.columns]

f = snakemake.input.mut
Mut = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut = Mut[Mut.columns.drop(list(Mut.filter(regex='Cetuximab')))]
Mut.columns = [c + "_mut" for c in Mut.columns]

target_col = snakemake.params.target_col
f = snakemake.input.response
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode target
Y_class_dict = {'PD': 0, 'SD': 1, 'OR': 1}
Y[target_col] = Y[target_col].replace(Y_class_dict)

df1 = pd.merge(Mut, CNV, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth, Expr, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
feature_col = all_df.columns.tolist()
all_df = all_df.select_dtypes([np.number])
all_df = pd.merge(all_df, Y[target_col],
                  right_index=True, left_index=True, how="right")

# fillna in features with median imputation
all_df[feature_col] = all_df[feature_col].    astype(
    float).apply(lambda col: col.fillna(col.median()))
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]

# scale features (0-1)
all_df = pd.DataFrame(MinMaxScaler().fit_transform(all_df.values),
                      columns=all_df.columns,
                      index=all_df.index)

# train-test split
train_models = Y[Y.is_test == False].index.unique()
test_models = Y[Y.is_test == True].index.unique()
X_train = all_df.loc[train_models, feature_col].values
y_train = all_df.loc[train_models, target_col].values
X_test = all_df.loc[test_models, feature_col].values
y_test = all_df.loc[test_models, target_col].values

# get indeces for feature subsets, one per OMIC
Meth_indeces = list(range(0, Meth.shape[1]))
pos = len(Meth_indeces)
Expr_indeces = list(range(Meth_indeces[-1]+1, pos + Expr.shape[1]))
pos += len(Expr_indeces)
Mut_indeces = list(range(Expr_indeces[-1]+1, pos + Mut.shape[1]))
pos += len(Mut_indeces)
CNV_indeces = list(range(Mut_indeces[-1]+1, pos + Mut.shape[1]))

model_filename = snakemake.input.model
classifier = pickle.load(open(model_filename, 'rb'))
# assess classifier performance on test set
grid_test_score = classifier.score(X_test, y_test)
y_pred = classifier.predict(X_test)
with open(snakemake.output.test_report, "w") as f:
    f.write(f'Accuracy on test set: {grid_test_score:.3f}')
    # print classification report on test set
    f.write(classification_report(y_test, y_pred,
                                  target_names=['PD', 'SD-OR']))

# return the marginal probability that the given sample has the label in question
y_test_predict_proba = classifier.predict_proba(X_test)

# plot Distributions of Predicted Probabilities of both classes
proba_df = pd.DataFrame(y_test_predict_proba,
                        columns=['PD', 'SD-OR'],
                        index=test_models).stack().reset_index()
proba_df.columns = ['ircc_id', 'class', 'proba']
proba_df.to_csv(snakemake.output.test_pred_proba, sep="\t")
# keep only prob for the positive class
proba_df = proba_df[proba_df['class'] == 'SD-OR'].drop('class', axis=1)

inv_Y_class_dict = {0: 'PD', 1: 'SD-OR'}
pred_df = pd.concat([pd.Series(l)
                     for l in [test_models, y_test, y_pred]], axis=1)
pred_df.columns = ["ircc_id", 'target', 'pred']
pred_df['pred'] = pred_df['pred'].replace(inv_Y_class_dict)
pred_df['target'] = pred_df['target'].replace(inv_Y_class_dict)
pred_df['correct'] = pred_df['target'] == pred_df['pred']
pred_df = pd.merge(pred_df, proba_df,
                   left_on=['ircc_id'],
                   right_on=['ircc_id'])

# plot the distribution of predition probs
fig, ax = plt.subplots(figsize=(6, 6))
plt.hist(pred_df[pred_df.target == 'PD'].proba, density=True, bins=list(np.arange(-1, 2, .1)),
         alpha=.5, color='red', label='PD')
plt.hist(pred_df[pred_df.target == 'SD-OR'].proba, density=True, bins=list(np.arange(-1, 2, .1)),
         alpha=.5, color='green',  label='SD-OR')
plt.axvline(.5, color='blue', linestyle='--', label='Boundary', linewidth=1)
plt.xlim([0, 1])
plt.title('Distribution of prediction probabilities', size=15)
plt.xlabel('Positive Probability', size=13)
plt.ylabel('Samples', size=13)
plt.legend(loc="upper left", prop={'size': 10})
plt.tight_layout()
fig.savefig(snakemake.output.test_pred_proba_plot,
            format='pdf',
            dpi=360)

# plot confusion matrix
ax = plot_confusion_matrix(classifier, X_test, y_test,
                           display_labels=['PD', 'SD-OR'],
                           cmap=plt.cm.Blues)
plt.tight_layout()
plt.savefig(snakemake.output.confusion_matrix_plot,
            format='pdf',
            dpi=360)

# plot ROC curve, ROC AUC
fp_rates, tp_rates, _ = roc_curve(y_test, y_test_predict_proba[:, 1])
roc_auc = auc(fp_rates, tp_rates)
fig, ax = plt.subplots(figsize=(6, 6))
plt.plot(fp_rates, tp_rates, color='green',
         lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
# plot decision point:
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = [i for i in cm.ravel()]
plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size=13)
plt.ylabel('True Positive Rate', size=13)
plt.title('ROC Curve', size=15)
plt.legend(loc="lower right", prop={'size': 10})
plt.subplots_adjust(wspace=.3)
plt.tight_layout()
fig.savefig(snakemake.output.auc_plot,
            format='pdf',
            dpi=360)

# plot feature coefficients for the meta classifier (LR)
coeff_plot_df = pd.DataFrame(classifier.meta_clf_.coef_.T,
                             columns=['SD-OR'],
                             index=['Meth', 'Expr', 'Mut', 'CNV'])
coeff_plot_df = coeff_plot_df.stack().reset_index()
coeff_plot_df.columns = ["omic", "class", "coeff"]
ax = sb.barplot(x="coeff",
                y="omic",
                color='g',
                data=coeff_plot_df)
plt.tight_layout()
fig.savefig(snakemake.output.meta_coeff_plot,
            format='pdf',
            dpi=360)

# plot RF feature importance by model (omic)
fig, axes = plt.subplots(2, 2, figsize=(10, 16))
axes = axes.flatten()
pipes = classifier.clfs_
for pipe, omic, ax in zip(pipes, ['Meth', 'Expr', 'Mut', 'CNV'], axes):
    # get names for selected features
    col_selector = pipe[0]
    level1_classifier = pipe[-1]
    params = level1_classifier.get_params()

    selected_indeces = pd.Series(col_selector.cols)
    for step in pipe[1:-1]:
        if hasattr(step, 'get_support'):
            selected_indeces = selected_indeces[step.get_support()]
        else:
            selected_indeces = selected_indeces[step.coef_]
    selected_indeces = selected_indeces.tolist()
    selected_features = [feature_col[i] for i in selected_indeces]

    coeff_plot_df = pd.DataFrame(level1_classifier.feature_importances_,
                                 columns=['SD-OR'],
                                 index=selected_features)
    coeff_plot_df = coeff_plot_df.stack().reset_index()
    coeff_plot_df.columns = ["feature", "class", "importance"]

    # select features w/t largest abs coeffs for viz
    df = coeff_plot_df.groupby('feature').importance.apply(
        pd.Series.max).        sort_values(ascending=False)
    # top = df.head(5).index.tolist()
    # bottom = df.tail(5).index.tolist()
    tokeep = df.head(15).index.tolist()
    plot_df = coeff_plot_df[coeff_plot_df.feature.isin(
        tokeep)].        sort_values('importance', ascending=False)

    ax = sb.barplot(x="importance",
                    y="feature",
                    color='b',
                    # hue="class",
                    # palette="Set2",
                    data=plot_df,
                    ax=ax)
    model = str(level1_classifier).split()[0]
    p = params['n_estimators']
    D = params['max_depth']
    C = params['criterion']
    n = len(selected_features)
    ax.set_title(
        f'{model} on {n} {omic} features; \n criterion: {C}; n_estimators: {p}; max_depth: {D}')
plt.subplots_adjust(wspace=0.7, hspace=0.5)
plt.tight_layout()
fig.savefig(snakemake.output.lv1_importance_plot,
            format='pdf',
            dpi=360)

# plot RF feature Pearson corr across models
tokeep = []
omic_labels = []
pipes = classifier.clfs_
omics = ['Meth', 'Expr', 'Mut', 'CNV']
n_feats = 5
for pipe, omic in zip(pipes, omics):
    # get names for selected features
    col_selector = pipe[0]
    level1_classifier = pipe[-1]
    params = level1_classifier.get_params()

    selected_indeces = pd.Series(col_selector.cols)
    for step in pipe[1:-1]:
        if hasattr(step, 'get_support'):
            selected_indeces = selected_indeces[step.get_support()]
        else:
            selected_indeces = selected_indeces[step.coef_]
    selected_indeces = selected_indeces.tolist()
    selected_features = [feature_col[i] for i in selected_indeces]

    coeff_plot_df = pd.DataFrame(level1_classifier.feature_importances_,
                                 columns=['SD-OR'],
                                 index=selected_features)
    coeff_plot_df = coeff_plot_df.stack().reset_index()
    coeff_plot_df.columns = ["feature", "class", "importance"]

    # select features w/t largest abs coeffs for viz
    df = coeff_plot_df.groupby('feature').importance.apply(
        pd.Series.max).        sort_values(ascending=False)
    tokeep.extend(df.head(n_feats).index.tolist())
    omic_labels.extend([omic]*n_feats)
corr_df = all_df[tokeep].corr()

lut = dict(zip(omics, sb.color_palette("husl", 4)))
row_colors = [lut[o] for o in omic_labels]
np.fill_diagonal(corr_df.values, 0)
g = sb.clustermap(corr_df,
                  row_colors=row_colors,
                  col_colors=row_colors,
                  cmap=sb.diverging_palette(20, 220, n=200))
fig = g.fig
legend_elements = [Patch(facecolor=lut[l]) for l in omics]
plt.figlegend(legend_elements, omics, loc='upper right', fontsize=12)
g.savefig(snakemake.output.feature_corr_plot,
          format='pdf',
          dpi=360)

pred_df = pd.concat([pd.Series(arr) for arr in [y_test,
                                                y_pred, y_test_predict_proba[:, 1]]],
                    axis=1)
pred_df.columns = ["y", "pred", "prob_SD-OR"]
pred_df["ircc_id"] = all_df.loc[test_models, target_col].index

pred_df["correct"] = pred_df.y == pred_df.pred
pred_df["pseudo_residual"] = np.abs(pred_df.y - pred_df["prob_SD-OR"])
# wrong predictions on test
# sorted by model confidence in the wrong pred
wrong_df = pred_df[pred_df.correct == False].sort_values(
    'pseudo_residual', ascending=False)
# right predictions on test sorted by confidence
right_df = pred_df[pred_df.correct == True].sort_values('pseudo_residual')
# add prediction labels
Y_val_dict = Y_class_dict = {0: 'PD', 1: 'SD-OR'}
pred_df["y_label"] = pred_df.y.replace(Y_val_dict)
pred_df["pred_label"] = pred_df.pred.replace(Y_val_dict)
pred_df.to_csv(snakemake.output.test_pred_tab, sep="\t")
wrong_PD_preds = pred_df[(pred_df.y_label == 'SD-OR')
                         & (pred_df.pred_label == 'PD')]

# Create a lime explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train,
                                                   training_labels=y_train,
                                                   categorical_features=Mut_indeces + CNV_indeces,
                                                   feature_selection="lasso_path",
                                                   class_names=["PD", "SD-OR"],
                                                   feature_names=feature_col,
                                                   discretize_continuous=False)
explain_outdir = snakemake.output.explain_plot_dir + "/"
os.makedirs(explain_outdir)
def predict_fn(x): return classifier.predict_proba(x).astype(float)
# explain top wrong / right prediction with LIME


def lime_ExplPlot(ix, df, id):
    # Display the predicted and true value for the wrong instance
    pred = Y_val_dict[df.loc[ix].pred]
    y = Y_val_dict[df.loc[ix].y]

    # Explanation for wrong prediction
    wrong = X_test[wrong_ix]  # wrong instance
    exp = explainer.explain_instance(data_row=wrong,
                                     predict_fn=predict_fn,
                                     num_features=20)
    # Plot the prediction explaination
    fig = exp.as_pyplot_figure()
    plt.title(f'{id} P:{pred}; Y:{y} explanation', size=12)
    plt.xlabel('effect on prediction', size=12)
    plt.tight_layout()
    fig.savefig(explain_outdir + f"LIME_barplot_{id}.pdf", format='pdf')

    # save html viz to file
    exp.save_to_file(explain_outdir + f'LIME_explanation_{id}.html')


pred_df[(pred_df.y_label == 'SD-OR') & (pred_df.pred_label == 'PD')]


for wrong_ix in wrong_PD_preds.index:
    # get test matrix row, sample id for best & worst ix'th prediction
    wrong = X_test[wrong_ix]  # wrong instance
    wrong_id = wrong_df.ircc_id[wrong_ix]  # wrong instance sample id

    lime_ExplPlot(wrong_ix, wrong_df, wrong_id)


for ix in range(0, 5):
    # get test matrix row, sample id for best & worst ix'th prediction
    wrong_ix = wrong_df.index[ix]  # index of the wrong instance in X_test
    wrong = X_test[wrong_ix]  # wrong instance
    wrong_id = wrong_df.ircc_id[wrong_df.index[ix]]  # wrong instance sample id
    right_ix = right_df.index[ix]
    right = X_test[right_df.index[ix]]
    right_id = right_df.ircc_id[right_df.index[ix]]

    lime_ExplPlot(wrong_ix, wrong_df, wrong_id)
    lime_ExplPlot(right_ix, right_df, right_id)

# clsfs_ contains the FITTED classifiers
# classifiers contains the original NON-FITTED classifiers (if use_clones == True (default))v
# print all decision trees for level 1 classifiers
for omic, ix_omic in zip(['Meth', 'Mut', 'CNV'], [0, 2, 3]):
    ix = 0
    pipe = classifier.clfs_[ix_omic]
    # get names for selected features
    col_selector = pipe[0]
    level1_classifier = pipe[-1]
    params = level1_classifier.get_params()

    selected_indeces = pd.Series(col_selector.cols)
    for step in pipe[1:-1]:
        if hasattr(step, 'get_support'):
            selected_indeces = selected_indeces[step.get_support()]
        else:
            selected_indeces = selected_indeces[step.coef_]
    selected_indeces = selected_indeces.tolist()
    selected_features = [feature_col[i] for i in selected_indeces]

    for single_tree in classifier.clfs_[ix_omic][-1].estimators_:
        dot_data = export_graphviz(single_tree,
                                   feature_names=selected_features,
                                   class_names=['PD', 'OR-SD'],
                                   filled=True, impurity=True,
                                   rounded=True)

        graph = graphviz.Source(dot_data, format='png')
        graph.render(
            explain_outdir + f'mut_featureEngineering_RFC_estimatorGraph_{omic}{ix}', format='pdf')
        ix += 1

shap.initjs()

# plot SHAP feature importance by model (omic)
pipes = classifier.clfs_

# get test matrix row, sample id for best & worst ix'th prediction
ix = 0
wrong_ix = wrong_df.index[ix]
wrong_id = wrong_df.ircc_id[wrong_ix]
right_ix = right_df.index[ix]
right_id = right_df.ircc_id[right_ix]


for pipe, omic, ax in zip(pipes, ['Meth', 'Expr', 'Mut', 'CNV'], axes):
    # get names for selected features
    col_selector = pipe[0]
    level1_classifier = pipe[-1]
    params = level1_classifier.get_params()

    selected_indeces = pd.Series(col_selector.cols)
    for step in pipe[1:-1]:
        if hasattr(step, 'get_support'):
            selected_indeces = selected_indeces[step.get_support()]
        else:
            selected_indeces = selected_indeces[step.coef_]
    selected_indeces = selected_indeces.tolist()
    selected_features = [feature_col[i] for i in selected_indeces]

    # get sliced test set
    selected_X_test = X_test[:, selected_indeces]

    # build shap explainer
    explainer = shap.explainers.Tree(level1_classifier,
                                     data=selected_X_test,
                                     model_output='raw',
                                     feature_perturbation='interventional',
                                     class_names=["PD", "OR-SD"],
                                     feature_names=selected_features)
    shap_values = explainer.shap_values(selected_X_test)

    # plot entropy-base feature importance
    plt.clf()
    shap.summary_plot(shap_values[1],
                      selected_X_test, feature_names=selected_features,
                      # class_names=["PD", "OR-SD"],
                      plot_type="bar")
    plt.savefig(explain_outdir + f"SHAP_RF_featureImportance_barplot_{omic}.png",
                format='png', dpi=720, bbox_inches='tight')

    # force plot for top right/wrong predictions (by prob)
    pred = Y_val_dict[wrong_df.loc[wrong_ix].pred]
    y = Y_val_dict[wrong_df.loc[wrong_ix].y]

    plt.clf()
    shap.force_plot(explainer.expected_value[1],
                    shap_values[1][wrong_ix],
                    selected_X_test[wrong_ix],
                    # class_names=["PD", "OR-SD"],
                    feature_names=selected_features,
                    matplotlib=True, show=False)
    plt.title(f'{wrong_id} P:{pred}; Y:{y}', size=12)
    plt.savefig(explain_outdir + f"SHAP_RF_ForcePlot_{wrong_id}_{omic}.pdf",
                format='pdf', dpi=720, bbox_inches='tight')

    pred = Y_val_dict[right_df.loc[right_ix].pred]
    y = Y_val_dict[right_df.loc[right_ix].y]

    plt.clf()
    shap.force_plot(explainer.expected_value[1],
                    shap_values[1][right_ix],
                    selected_X_test[right_ix],
                    # class_names=["PD", "OR-SD"],
                    feature_names=selected_features,
                    matplotlib=True, show=False)
    plt.title(f'{right_id} P:{pred}; Y:{y}', size=12)
    plt.savefig(explain_outdir + f"SHAP_RF_ForcePlot_{right_id}_{omic}.pdf",
                format='pdf', dpi=720, bbox_inches='tight')


# plot SHAP feature importance by model (omic)
pipes = classifier.clfs_

# get test matrix row, sample id for best & worst ix'th prediction
wrong_ix1, wrong_ix2 = wrong_PD_preds.index
wrong_id1 = wrong_df.ircc_id[wrong_ix1]
wrong_id2 = wrong_df.ircc_id[wrong_ix2]

for pipe, omic in zip(pipes, ['Meth', 'Expr', 'Mut', 'CNV']):
    # get names for selected features
    col_selector = pipe[0]
    level1_classifier = pipe[-1]
    params = level1_classifier.get_params()

    selected_indeces = pd.Series(col_selector.cols)
    for step in pipe[1:-1]:
        if hasattr(step, 'get_support'):
            selected_indeces = selected_indeces[step.get_support()]
        else:
            selected_indeces = selected_indeces[step.coef_]
    selected_indeces = selected_indeces.tolist()
    selected_features = [feature_col[i] for i in selected_indeces]

    # get sliced test set
    selected_X_test = X_test[:, selected_indeces]

    # build shap explainer
    explainer = shap.explainers.Tree(level1_classifier,
                                     data=selected_X_test,
                                     model_output='raw',
                                     feature_perturbation='interventional',
                                     class_names=["PD", "OR-SD"],
                                     feature_names=selected_features)
    shap_values = explainer.shap_values(selected_X_test)

    # plot entropy-base feature importance
    plt.clf()
    shap.summary_plot(shap_values[1],
                      selected_X_test,
                      feature_names=selected_features,
                      plot_type="bar")
    plt.savefig(explain_outdir + f"SHAP_RF_featureImportance_barplot_{omic}.png",
                format='png', dpi=720, bbox_inches='tight')

    # force plot for top right/wrong predictions (by prob)
    pred = Y_val_dict[wrong_df.loc[wrong_ix1].pred]
    y = Y_val_dict[wrong_df.loc[wrong_ix1].y]

    plt.clf()
    shap.force_plot(explainer.expected_value[1],
                    shap_values[1][wrong_ix1],
                    selected_X_test[wrong_ix1],
                    # class_names=["PD", "OR-SD"],
                    feature_names=selected_features,
                    matplotlib=True, show=False)
    plt.title(f'{wrong_id1} P:{pred}; Y:{y}', size=12)
    plt.savefig(explain_outdir + f"SHAP_RF_ForcePlot_{wrong_id1}_{omic}.pdf",
                format='pdf', dpi=720, bbox_inches='tight')

    pred = Y_val_dict[wrong_df.loc[wrong_ix1].pred]
    y = Y_val_dict[wrong_df.loc[wrong_ix1].y]

    plt.clf()
    shap.force_plot(explainer.expected_value[1],
                    shap_values[1][wrong_ix1],
                    selected_X_test[wrong_ix1],
                    # class_names=["PD", "OR-SD"],
                    feature_names=selected_features,
                    matplotlib=True, show=False)
    plt.title(f'{wrong_id2} P:{pred}; Y:{y}', size=12)
    plt.savefig(explain_outdir + f"SHAP_RF_ForcePlot_{wrong_id2}_{omic}.pdf",
                format='pdf', dpi=720, bbox_inches='tight')
