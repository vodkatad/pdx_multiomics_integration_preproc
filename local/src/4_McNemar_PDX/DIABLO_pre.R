library(glmnet)
library(data.table)

# load all omic input data
f <- snakemake@input[["expr"]]
expr <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(expr)<-expr$ircc_id
expr<-subset(expr, select = -c(ircc_id))
print("loaded expr")
f <- snakemake@input[["meth"]]
meth <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(meth)<-meth$ircc_id
meth<-subset(meth, select = -c(ircc_id))
print("loaded meth")
f <- snakemake@input[["cnv"]]
cnv <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(cnv)<-cnv$ircc_id
cnv<-subset(cnv, select = -c(ircc_id))
print("loaded cnv")
f <- snakemake@input[["mut"]]
mut <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(mut)<-mut$ircc_id
mut<-subset(mut, select = -c(ircc_id))
print("loaded mut")
f <- snakemake@input[["clin"]]
clin <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(clin)<-clin$ircc_id
mut<-subset(clin, select = -c(ircc_id))
print("loaded clin")
# load target
f <- snakemake@input[["response"]]
response<-as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(response)<-response$ircc_id
target_col<-quote(snakemake@params[["target_col"]])
print("loaded Y")

# apply train/test split (already stratified)
test_models<-rownames(response)[
	response$is_test==TRUE]
train_models<-rownames(response)[
	response$is_test==FALSE]
Y_test<-factor(response[match(
	test_models,rownames(response)),eval(target_col)])
expr_test<-expr[match(test_models,rownames(expr)),]
meth_test<-meth[match(test_models,rownames(meth)),]
mut_test<-mut[match(test_models,rownames(mut)),]
clin_test<-clin[match(test_models,rownames(clin)),]
cnv_test<-cnv[match(test_models,rownames(cnv)),]

Y_train<-factor(response[match(
	train_models,rownames(response)),eval(target_col)])
expr_train<-expr[match(train_models,rownames(expr)),]
meth_train<-meth[match(train_models,rownames(meth)),]
mut_train<-mut[match(train_models,rownames(mut)),]
clin_train<-clin[match(train_models,rownames(clin)),]
cnv_train<-cnv[match(train_models,rownames(cnv)),]


write.table(Y_test, 
	file = snakemake@output[["Y_test"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
write.table(Y_train, 
	file = snakemake@output[["Y_train"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)

write.table(cnv_test, 
	file = snakemake@output[["cnv_test"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
write.table(cnv_train, 
	file = snakemake@output[["cnv_train"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
write.table(mut_test, 
	file = snakemake@output[["mut_test"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
write.table(mut_train, 
	file = snakemake@output[["mut_train"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
write.table(clin_test, 
	file = snakemake@output[["clin_test"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
write.table(clin_train, 
	file = snakemake@output[["clin_train"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
print("written Y, cnv, mutm, clin train / test ")


# add median imputation for any missing value
# impute separately for train and test
for(i in 1:ncol(expr_train)){
	expr_train[,i][is.na(expr_train[,i])]<-median(expr_train[,i],na.rm=TRUE)}
for(i in 1:ncol(expr_test)){
	expr_test[,i][is.na(expr_test[,i])]<-median(expr_test[,i],na.rm=TRUE)}
for(i in 1:ncol(meth_train)){
	meth_train[,i][is.na(meth_train[,i])]<-median(meth_train[,i],na.rm=TRUE)}
for(i in 1:ncol(meth_test)){
	meth_test[,i][is.na(meth_test[,i])]<-median(meth_test[,i],na.rm=TRUE)}
# Lasso supervised feature selection on expression, methylation training data
lasso_fit <- cv.glmnet(as.matrix(expr_train), Y_train, 
	family = "binomial", alpha = 1)
coef <- predict(lasso_fit, 
	s = "lambda.min", type = "nonzero")
result_expr <- data.frame(GENE = names(as.matrix(coef(lasso_fit, s = "lambda.min"))
                                [as.matrix(coef(lasso_fit, s = "lambda.min"))[,1]!=0, 1])[-1], 
                   SCORE = as.numeric(as.matrix(coef(lasso_fit, s = "lambda.min"))
                                      [as.matrix(coef(lasso_fit, 
                                                      s = "lambda.min"))[,1]!=0, 1])[-1])
result_expr <- result_expr[order(-abs(result_expr$SCORE)),]
# log expr coeffs
write.table(result_expr, 
	file = snakemake@log[["expr_coef"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
# slice selected features 
expr_train <- subset(expr_train, select = as.character(result_expr$GENE))
expr_test<-subset(expr_test,select=colnames(expr_train))

# repeat for methylation data
lasso_fit <- cv.glmnet(as.matrix(meth_train), Y_train, 
	family = "binomial", alpha = 1)
coef <- predict(lasso_fit, 
	s = "lambda.min", type = "nonzero")
result_meth <- data.frame(GENE = names(as.matrix(coef(lasso_fit, s = "lambda.min"))
                                [as.matrix(coef(lasso_fit, s = "lambda.min"))[,1]!=0, 1])[-1], 
                   SCORE = as.numeric(as.matrix(coef(lasso_fit, s = "lambda.min"))
                                      [as.matrix(coef(lasso_fit, 
                                                      s = "lambda.min"))[,1]!=0, 1])[-1])
result_meth <- result_meth[order(-abs(result_meth$SCORE)),]
write.table(result_meth, 
	file = snakemake@log[["meth_coef"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
meth_train <- subset(meth_train, select = as.character(result_meth$GENE))
meth_test<-subset(expr_test,select=colnames(meth_train))

# write all train and test datasets
write.table(expr_test, 
	file = snakemake@output[["expr_test"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
write.table(expr_train, 
	file = snakemake@output[["expr_train"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)

write.table(meth_test, 
	file = snakemake@output[["meth_test"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)
write.table(meth_train, 
	file = snakemake@output[["meth_train"]], 
	append = FALSE, quote = FALSE , sep = "\t",
	eol = "\n", na = "NA", dec = ".", 
	row.names = TRUE, col.names = TRUE)