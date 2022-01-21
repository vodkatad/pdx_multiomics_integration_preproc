# https://en.wikipedia.org/wiki/McNemar%27s_test
# http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/#example-2-mcnemars-test-for-scenario-b
# McNemar's Test [McNemar, Quinn, 1947. "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika. 12 (2): 153–157.]
#  (sometimes also called "within-subjects chi-squared test") is a statistical 
# test for paired nominal data. In context of machine learning (or statistical) models, 
# we can use McNemar's Test to compare the predictive accuracy of two models. 
# McNemar's test is based on a 2 times 2 contigency table of the two model's predictions.


import numpy as np
import pandas as pd
import pickle
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table

def McNemar_test(y_target, y_model1, y_model2):
	contingency_tab = mcnemar_table(y_target=y_target, 
			y_model1=y_model1, 
			y_model2=y_model2)
	chi2, p = mcnemar(ary=contingency_tab, exact=True)
	return chi2, p
# load triple negative feature
tripleNeg_df_df = pd.read_csv(snakemake.input.tripleNeg, 
			sep='\t', header=0, index_col=0)['KRAS_BRAF_NRAS_triple_neg']

tab_arr = []
split_index = 0
models_compared = snakemake.params.model_name + '__vs__' + snakemake.params.benchmark_name
for trio in zip(snakemake.input.models,
		snakemake.input.X, 
		snakemake.input.Y):
	modelFile, XtestFile, YtestFile  = trio
	target_col = snakemake.params.target_col
	# load the model from file
	stacked_classifier = pickle.load(open(modelFile, 'rb'))
	# load test set
	X_test = pd.read_csv(XtestFile, sep="\t", header=0, index_col=0)
	y_test = pd.read_csv(YtestFile, sep="\t", header=0, index_col=0)[target_col]
	# assess best classifier performance on test set
	grid_test_score = stacked_classifier.score(X_test, y_test)
	stacked_y_pred = stacked_classifier.predict(X_test)
	# convert to readable labels
	Y_val_dict = Y_class_dict={0:'PD', 1:'SD-OR'}
	stacked_y_test_readable = [Y_class_dict[y] for y in y_test]
	stacked_y_pred_readable = [Y_class_dict[y] for y in stacked_y_pred]

	# predict based on current gold-standard diagnostic signature
	# aka KRAS-NRAS-BRAF triple negative (triple wild type) status
	# if triple negative (1) -> predict as responsive to cetuximab (1)
	tripleNeg_y_pred = tripleNeg_df_df.loc[X_test.index]
	tripleNeg_y_pred_readable = [Y_class_dict[y] for y in tripleNeg_y_pred] 

	chi2,p = McNemar_test(y_test, stacked_y_pred, tripleNeg_y_pred)

	tab_arr.append([models_compared, split_index, chi2, p])	
	split_index += 1

out_tab = pd.DataFrame(tab_arr, columns=["models_compared", 
					"split_index", "McNemar_Chi2",
					"McNemar_Pval"])
out_tab["McNemar_Pval_BonfAdj"] = out_tab.McNemar_Pval * len(out_tab) 
out_tab.to_csv(snakemake.output.test_tab, sep='\t')