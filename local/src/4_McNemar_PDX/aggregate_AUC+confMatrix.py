
import seaborn as sb
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import distance

from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
# Hyperparameter tuning

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

# load triple negative feature
tripleNeg_df = pd.read_csv(snakemake.input.tripleNeg,
                           sep='\t', header=0, index_col=0)['KRAS_BRAF_NRAS_triple_neg']
target_col = snakemake.params.target_col
Y_class_dict = {0: 'PD', 1: 'SD-OR'}


def evaluate_model(modelFile, XtestFile, YtestFile, model_name):
    # load the model from file
    classifier = pickle.load(open(modelFile, 'rb'))
    # load test set
    X_test = pd.read_csv(XtestFile, sep="\t", header=0, index_col=0)
    y_test = pd.read_csv(YtestFile, sep="\t", header=0,
                         index_col=0)[target_col]
    # assess best classifier performance on test set (accuracy)
    grid_test_score = classifier.score(X_test, y_test)
    y_pred = classifier.predict(X_test)
    # convert to readable labels
    y_test_readable = [Y_class_dict[y] for y in y_test]
    y_pred_readable = [Y_class_dict[y] for y in y_pred]

    # compute and flatten confusion matrix
    confMatrix = confusion_matrix(y_test_readable,
                                  y_pred_readable,
                                  labels=['PD', 'SD-OR'])
    confusion_df = pd.DataFrame(confMatrix,
                                columns=['pred_PD', 'pred_SD-OR'],
                                index=['true_PD', 'true_SD-OR']).stack()
    confusion_df.index = ['__'.join(ix) for ix in confusion_df.index]
    # calc model AUC score on test set
    y_test_predict_proba = classifier.predict_proba(X_test)
    AUC = roc_auc_score(y_test, y_test_predict_proba[:, 1])

    # evaluate model perfomance on clinically relevant subgroup of models
    # aka KRAS-BRAF-NRAS triple negative;
    # merge KRAS-NRAS-BRAF predictions, stacked prediction, truth
    pred_df = pd.DataFrame.from_dict({'Y': y_test,
                                      'classifier': y_pred,
                                      'tripleNeg': tripleNeg_df.loc[y_test.index]})
    # how many additional (compared to the KRAS-BRAF-NRAS signature) tripleNeg==1
    # models do we correctly categorise as non-responders (aka PD)?
    tripleNegResistAll_count = pred_df[(pred_df.Y == 0) & (
        pred_df.tripleNeg == 1)].index.nunique()
    tripleNegResistPredicted_count = pred_df[(pred_df.Y == 0) & (pred_df.tripleNeg == 1) &
                                             (pred_df.classifier == 0)].index.nunique()
    ResistPredicted_count = pred_df[(pred_df.Y == 0) & (
        pred_df.tripleNeg == 0)].index.nunique()
    additionalResist_ratio = tripleNegResistPredicted_count / \
        ResistPredicted_count  # new predicted / previously predicted
    # how many additional (compared to the KRAS-BRAF-NRAS signature) tripleNeg==0 models
    # do we correctly categorise as responders (aka SD-OR)?
    tripleNegRespondAll_count = pred_df[(pred_df.Y == 1) & (
        pred_df.tripleNeg == 0)].index.nunique()
    tripleNegRespondPredicted_count = pred_df[(pred_df.Y == 1) & (pred_df.tripleNeg == 0) &
                                              (pred_df.classifier == 1)].index.nunique()
    RespondPredicted_count = pred_df[(pred_df.Y == 1) & (
        pred_df.tripleNeg == 1)].index.nunique()
    additionalRespond_ratio = tripleNegRespondPredicted_count / \
        RespondPredicted_count  # new predicted / previously predicted

    return [model_name,
            grid_test_score,
            AUC,
            additionalResist_ratio,
            tripleNegResistPredicted_count,
            additionalRespond_ratio,
            tripleNegRespondPredicted_count] + confusion_df.tolist()


def evaluate_tripleNeg(XtestFile, YtestFile):
    # load test set
    X_test = pd.read_csv(XtestFile, sep="\t", header=0, index_col=0)
    y_test = pd.read_csv(YtestFile, sep="\t", header=0,
                         index_col=0)[target_col]
    y_test_readable = [Y_class_dict[y] for y in y_test]
    # predict Cetuximab sensitivity based on current gold-standard diagnostic signature
    # aka KRAS-NRAS-BRAF triple negative (triple wild type) status
    # if triple negative (1) -> predict as responsive to cetuximab (1)
    tripleNeg_y_pred = tripleNeg_df.loc[X_test.index]
    tripleNeg_y_pred_readable = [Y_class_dict[y] for y in tripleNeg_y_pred]
    # compute accuracy on test set
    test_accu = 1 - \
        distance.cityblock(y_test, tripleNeg_y_pred) / len(y_test)
    # compute triple neg confusion matrix
    tripleNeg_confMatrix = confusion_matrix(y_test_readable,
                                            tripleNeg_y_pred_readable,
                                            labels=['PD', 'SD-OR'])
    tripleNeg_confusion_df = pd.DataFrame(tripleNeg_confMatrix,
                                          columns=['pred_PD', 'pred_SD-OR'],
                                          index=['true_PD', 'true_SD-OR']).stack()
    tripleNeg_confusion_df.index = [
        '__'.join(ix) for ix in tripleNeg_confusion_df.index]
    # compute triple neg AUC
    tripleNeg_y_test_predict_proba = tripleNeg_y_pred
    tripleNeg_AUC = roc_auc_score(y_test, tripleNeg_y_test_predict_proba)
    return ['KRAS_BRAF_NRAS_triple_neg', test_accu,
            tripleNeg_AUC, np.nan, np.nan, np.nan, np.nan] + tripleNeg_confusion_df.tolist()


def evaluate_DIABLO(DIABLOPredFile, YtestFile):
    DIABLOPred = pd.read_csv(DIABLOPredFile, sep='\t',
                             header=None, index_col=0)[1]
    y_test = pd.read_csv(YtestFile, sep="\t", header=0,
                         index_col=0)[target_col]
    # convert predictions to readable class labels, handle nans
    DIABLOPred = DIABLOPred.loc[y_test.index].fillna(0)
    DIABLOPred_readable = [Y_class_dict[y] for y in DIABLOPred]
    y_test_readable = [Y_class_dict[y] for y in y_test]
    # compute accuracy on test set
    test_accu = 1 - \
        distance.cityblock(y_test, DIABLOPred) / len(y_test)
    # compute DIABLO confusion matrix
    confMatrix = confusion_matrix(y_test_readable,
                                  DIABLOPred_readable,
                                  labels=['PD', 'SD-OR'])
    confMatrix_df = pd.DataFrame(confMatrix,
                                 columns=['pred_PD', 'pred_SD-OR'],
                                 index=['true_PD', 'true_SD-OR']).stack()
    confMatrix_df.index = [
        '__'.join(ix) for ix in confMatrix_df.index]
    # use binary prediction as prediction proba for AUC calc
    predict_proba = DIABLOPred
    DIABLO_AUC = roc_auc_score(y_test, predict_proba)

    # evaluate model perfomance on clinically relevant subgroup of models
    # aka KRAS-BRAF-NRAS triple negative;
    # merge KRAS-NRAS-BRAF predictions, DIABLO prediction, truth
    pred_df = pd.DataFrame.from_dict({'Y': y_test,
                                      'classifier': DIABLOPred,
                                      'tripleNeg': tripleNeg_df.loc[y_test.index]})
    # how many additional (compared to the KRAS-BRAF-NRAS signature) tripleNeg==1
    # models do we correctly categorise as non-responders (aka PD)?
    tripleNegResistAll_count = pred_df[(pred_df.Y == 0) & (
        pred_df.tripleNeg == 1)].index.nunique()
    tripleNegResistPredicted_count = pred_df[(pred_df.Y == 0) & (pred_df.tripleNeg == 1) &
                                             (pred_df.classifier == 0)].index.nunique()
    ResistPredicted_count = pred_df[(pred_df.Y == 0) & (
        pred_df.tripleNeg == 0)].index.nunique()
    additionalResist_ratio = tripleNegResistPredicted_count / \
        ResistPredicted_count  # new predicted / previously predicted
    # how many additional (compared to the KRAS-BRAF-NRAS signature) tripleNeg==0 models
    # do we correctly categorise as responders (aka SD-OR)?
    tripleNegRespondAll_count = pred_df[(pred_df.Y == 1) & (
        pred_df.tripleNeg == 0)].index.nunique()
    tripleNegRespondPredicted_count = pred_df[(pred_df.Y == 1) & (pred_df.tripleNeg == 0) &
                                              (pred_df.classifier == 1)].index.nunique()
    RespondPredicted_count = pred_df[(pred_df.Y == 1) & (
        pred_df.tripleNeg == 1)].index.nunique()
    additionalRespond_ratio = tripleNegRespondPredicted_count / \
        RespondPredicted_count  # new predicted / previously predicted

    return ['DIABLO_sPLS-DA',
            test_accu,
            DIABLO_AUC,
            additionalResist_ratio,
            tripleNegResistPredicted_count,
            additionalRespond_ratio,
            tripleNegRespondPredicted_count] + confMatrix_df.tolist()


tab_arr = []
split_index = 0
for tupla in zip(snakemake.input.stacked_models,
                 snakemake.input.rawL1elasticnet_models,
                 snakemake.input.handpickedElasticnet,
                 snakemake.input.KNBrasElasticnet,
                 snakemake.input.DIABLOsPLSDA_pred,
                 snakemake.input.tripleNegResample_stacked,
                 snakemake.input.X_eng,
                 snakemake.input.X_raw,
                 snakemake.input.X_handpicked,
                 snakemake.input.X_KNBras,
                 snakemake.input.Y):
    stackedFile, elasticFile, handpickedElastiFile,\
         KNBrasElasticFile, DIABLOPredFile, \
        resampledStackedFile, XtestFile_eng, \
        XtestFile_raw, XtestFile_handpicked, \
            XtestFile_KNBras, YtestFile = tupla
    tab_arr.append([split_index] +
                   evaluate_model(stackedFile,
                                  XtestFile_eng, YtestFile, 'stackedCVClassifier'))
    tab_arr.append([split_index] +
                   evaluate_model(handpickedElastiFile,
                                  XtestFile_handpicked, YtestFile, 'handpickedElasticnet'))
    tab_arr.append([split_index] +
                   evaluate_model(KNBrasElasticFile,
                                  XtestFile_KNBras, YtestFile, 'KNBrasElasticnet'))
    tab_arr.append([split_index] +
                   evaluate_model(elasticFile,
                                  XtestFile_raw, YtestFile, 'rawL1elasticnet'))
    tab_arr.append([split_index] +
                   evaluate_model(resampledStackedFile,
                                  XtestFile_eng, YtestFile, 'tripleNegResample_stackedCVClassifier'))
    tab_arr.append([split_index] +
                   evaluate_tripleNeg(XtestFile_eng, YtestFile))
    tab_arr.append([split_index] +
                   evaluate_DIABLO(DIABLOPredFile, YtestFile))
    split_index += 1
out_tab = pd.DataFrame(tab_arr, columns=["split_index", 'model_name',
                                         "grid_test_accu",
                                         "AUC",
                                         "additionalResistVStripleNeg_ratio",
                                         "additionalResistVStripleNeg_count",
                                         'additionalRespondVStripleNeg_ratio',
                                         "additionalRespondVStripleNeg_count"] + ['true_PD__pred_PD',
                                                                                  'true_PD__pred_SD-OR',
                                                                                  'true_SD-OR__pred_PD',
                                                                                  'true_SD-OR__pred_SD-OR'])
out_tab.to_csv(snakemake.output.performance_tab, sep='\t')
