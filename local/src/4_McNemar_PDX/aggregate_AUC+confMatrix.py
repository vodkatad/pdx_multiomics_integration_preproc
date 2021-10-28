
import numpy as np
import pandas as pd
import pickle
	
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
# Hyperparameter tuning

# Visualizations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# Set default font size
plt.rcParams['font.size'] = 24
import seaborn as sb
# Set default font size
sb.set(font_scale = .8)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)

# load triple negative feature
tripleNeg_df_df = pd.read_csv(snakemake.input.tripleNeg, 
			sep='\t', header=0, index_col=0)['KRAS_BRAF_NRAS_triple_neg']

tab_arr = []
split_index = 0
for trio in zip(snakemake.input.models,
		snakemake.input.X, 
		snakemake.input.Y):
	modelFile, XtestFile, YtestFile  = trio
	target_col = snakemake.params.target_col
	# load the model from file
	classifier = pickle.load(open(modelFile, 'rb'))
	# load test set
	X_test = pd.read_csv(XtestFile, sep="\t", header=0, index_col=0)
	y_test = pd.read_csv(YtestFile, sep="\t", header=0, index_col=0)[target_col]
	# assess best classifier performance on test set
	grid_test_score = classifier.score(X_test, y_test)
	y_pred = classifier.predict(X_test)
	# convert to readable labels
	Y_val_dict = Y_class_dict={0:'PD', 1:'SD-OR'}
	y_test_readable = [Y_class_dict[y] for y in y_test]
	y_pred_readable = [Y_class_dict[y] for y in y_pred]
	## print classification report on test set
	#print(classification_report(y_test, y_pred, target_names=['PD', 'SD-OR']))

	# predict based on current gold-standard diagnostic signature
	# aka KRAS-NRAS-BRAF triple negative (triple wild type) status
	# if triple negative (1) -> predict as responsive to cetuximab (1)
	tripleNeg_y_pred = tripleNeg_df_df.loc[X_test.index]
	tripleNeg_y_pred_readable = [Y_class_dict[y] for y in tripleNeg_y_pred] 

	# compute and flatten confusion matrix		
	confMatrix = confusion_matrix(y_test_readable, 
							y_pred_readable, 
							labels=['PD', 'SD-OR'])
	confusion_df = pd.DataFrame(confMatrix, 
					columns=['pred_PD', 'pred_SD-OR'],
					index=['true_PD', 'true_SD-OR']).stack()
	confusion_df.index = ['__'.join(ix) for ix in confusion_df.index]
	# compute triple neg confusion matrix
	tripleNeg_confMatrix = confusion_matrix(y_test_readable, 
							tripleNeg_y_pred_readable, 
							labels=['PD', 'SD-OR'])
	tripleNeg_confusion_df = pd.DataFrame(tripleNeg_confMatrix, 
					columns=['pred_PD', 'pred_SD-OR'],
					index=['true_PD', 'true_SD-OR']).stack()
	tripleNeg_confusion_df.index = ['__'.join(ix) for ix in tripleNeg_confusion_df.index]

	# return the marginal probability according to the model 
	# that the given instance has the predicted label
	y_test_predict_proba = classifier.predict_proba(X_test)
	AUC = roc_auc_score(y_test, y_test_predict_proba[:, 1])
	# compute triple neg AUC
	tripleNeg_y_test_predict_proba = tripleNeg_y_pred
	tripleNeg_AUC = roc_auc_score(y_test, tripleNeg_y_test_predict_proba)

	tab_arr.append([snakemake.params.model_name, 
		split_index, grid_test_score, AUC] + confusion_df.tolist())
	tab_arr.append(['KRAS_BRAF_NRAS_triple_neg', 
		split_index, np.nan, tripleNeg_AUC] + tripleNeg_confusion_df.tolist())	
	split_index += 1
out_tab = pd.DataFrame(tab_arr, columns=["model_name", "split_index", 
					"grid_test_score", 
					"AUC"] + confusion_df.index.tolist())
out_tab.to_csv(snakemake.output.performance_tab, sep='\t')