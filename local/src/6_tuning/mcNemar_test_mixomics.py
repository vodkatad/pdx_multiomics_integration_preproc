# https://en.wikipedia.org/wiki/McNemar%27s_test
# http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/#example-2-mcnemars-test-for-scenario-b
# McNemar's Test [McNemar, Quinn, 1947. "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika. 12 (2): 153–157.]
#  (sometimes also called "within-subjects chi-squared test") is a statistical 
# test for paired nominal data. In context of machine learning (or statistical) models, 
# we can use McNemar's Test to compare the predictive accuracy of two models. 
# McNemar's test is based on a 2 times 2 contigency table of the two model's predictions.

import numpy as np
import pickle
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table

def McNemar_test(y_target, y_model1, y_model2):
	contingency_tab = mcnemar_table(y_target=y_target, 
			y_model1=y_model1, 
			y_model2=y_model2)
	chi2, p = mcnemar(ary=contingency_tab, exact=True)
	return chi2, p

# read mixomic confusion matrix
mixomics_confMatrix = pd.read_csv(snakemake.input.mixomics_confMatrix, 
				sep='\t',
				header=0,
				index_col=0)
conf_arr = mixomics_confMatrix.values
# count right and wrong predictions
mixomics_right = np.trace(conf_arr)
mixomics_wrong = np.sum(conf_arr) - mixomics_right 

tab_arr = []
split_index = 0
for quadrio in zip(snakemake.input.response, 
		snakemake.input.models,
		snakemake.input.X, 
		snakemake.input.Y):
	responseFile, modelFile, XtestFile, YtestFile  = quadrio
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

	# compute confusion matrix		
	confMatrix = confusion_matrix(y_test_readable, 
							y_pred_readable, 
							labels=['PD', 'SD-OR'])
	# count right and wrong predictions
	stacked_right = np.trace(confMatrix)
	stacked_wrong = np.sum(conf_aconfMatrixrr) - stacked_right
	# The mcnemar funtion expects a 2x2 contingency table as a NumPy array
	contingency_table = np.array([[mixomics_right, mixomics_wrong],
					[15, 15]])
