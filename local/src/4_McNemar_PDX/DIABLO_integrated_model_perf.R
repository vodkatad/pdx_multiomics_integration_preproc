# see: https://github.com/NikolayOskolkov/SupervisedOMICsIntegration/blob/master/supervised_omics_integr_CLL.Rmd
library("mixOmics")
require(data.table)

# load pre-tuned model
f <- snakemake@input[["model"]]
res<-readRDS(f)
# get number of components
ncomp<-res$ncomp[1]
print("loaded model")

# assess performance on train set (4-fold CV x 100 repeats) 
# with Weighted prediction
perf.res <- perf(res,
                 validation = "Mfold",
                 cpus=snakemake@threads,
                 folds = 4,
                 nrepeat = 5, 
                 dist = "mahalanobis.dist") 
print("checked model perf on training set")

# calc log balanced error rate on training set
perf_res_train<-as.data.frame(perf.res$WeightedVote.error.rate)
f <- snakemake@output[["trainBER_table"]]
write.table(perf_res_train, 
            file = f,
            quote = FALSE, 
            sep = "\t",
            row.names = TRUE,
            col.names = TRUE)
print("save train set BER")

# Low-dimensional latent PLS space representation of each individual OMIC
# samples are represented as points placed according to 
# their projection in the smaller subspace spanned by the components
f <- snakemake@output[["PlotIndiv"]]
pdf(f,
    width = 10, height = 6, bg = "white", 
    colormodel = "cmyk",paper = "A4") 
plotIndiv(res,legend=TRUE,
          title="PDx Omics sPLS-DA",
          ellipse=FALSE,
          ind.names=FALSE,
          cex=.5)
dev.off() 
print("plotted omic latent space viz")

# save loadings for each omic and each component
loadings<-as.data.frame(plotLoadings(res,
                             contrib='max',
                             method='median',
                             plot=FALSE))
f<-snakemake@output[["loadings_table"]]
write.table(loadings, 
            file = f,
            quote = FALSE,
            sep = "\t",
            row.names = TRUE,
            col.names = TRUE)
print("write omic;component loadings")

# diplay each individual PDX model on the plot of each of the OMICs
# against each other which are defined by the top loadings from 
# their respective components.
f<-snakemake@output[["plotDiablo"]]
pdf(f,
    width = 10, height = 6, bg = "white", 
    colormodel = "cmyk",paper = "A4")
for (c in seq(1, ncomp))
{
   plotDiablo(res,ncomp=c) 
}
dev.off()
print("plot DIABLO individual PDX in latent space")

# Display “arrow plot”  
# The start of the arrow indicates the location of the sample in X in one plot, 
# and the tip the location of the sample in Y in the other plot. 
# Short arrows indicate a strong agreement between the matching data sets, 
# long arrows a disagreement between the matching data sets.
# This plot highlights the agreement between all data sets at 
# the sample level, when modelled with DIABLO.
f<-snakemake@output[["plotArrow"]]
pdf(f,
    width = 8, height = 6, bg = "white",
    colormodel = "cmyk",paper = "A4") 
plotArrow(res,ind.names=FALSE,legend=TRUE,
          title="PDx omics integration arrow plot")
dev.off() 
print("plot DIABLO arrow")

# calculate “circos plot” that diaplays variable 
# correlation among different OMICs datasets
# the variables for this plot were selected simultaneously from all the OMICs, i.e. they are 
# not equavalent to those obtained from each individual OMIC separately.
f<-snakemake@output[["circosPlot"]]
pdf(f,
    width = 18, height = 16, bg = "white", 
    colormodel = "cmyk",paper = "A4") 
circosPlot(res,
           cutoff=.5,
           line=FALSE,
           size.variables=0.5)
dev.off()
print("plot circos")

# Plot the correlation heatmap showing strongly correlated blocks 
# of gene expression, methylation, mutation, clinical, CNV  features
# that provide clustering of individuals into responders/non-responders
f<-snakemake@output[["corrHeatmap"]]
pdf(f,
    width = 24, height = 14, bg = "white", 
    colormodel = "cmyk",paper = "A4") 
cimDiablo(res)
dev.off()
print("plot feature correlation heatmap")

### Predict target classes on test set,
# assess prediction accuracy

# load test data for e/a omic
f <- snakemake@input[["expr_Xtest"]]
expr_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(expr_test)<-expr_test$ircc_id
expr_test<-subset(expr_test, select = -c(ircc_id))

f <- snakemake@input[["meth_Xtest"]]
meth_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(meth_test)<-meth_test$ircc_id
meth_test<-subset(meth_test, select = -c(ircc_id))

f <- snakemake@input[["cnv_Xtest"]]
cnv_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(cnv_test)<-cnv_test$ircc_id
cnv_test<-subset(cnv_test, select = -c(ircc_id))

f <- snakemake@input[["mut_Xtest"]] 
mut_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(mut_test)<-mut_test$ircc_id
mut_test<-subset(mut_test, select = -c(ircc_id))

f <- snakemake@input[["clin_Xtest"]]
clin_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(clin_test)<-clin_test$ircc_id
clin_test<-subset(clin_test, select = -c(ircc_id))

# load target
f <- snakemake@input[["Ytest"]] 
Y_test<-as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(Y_test)<-Y_test$ircc_id
Y_test<-subset(Y_test, select = -c(ircc_id))
target_col<-quote(snakemake@params[["target_col"]])
Y_test<-factor(Y_test[,eval(target_col)])
print("loaded testfiles")

# build test set 
data.test<-list(expr=expr_test,
                mut=mut_test,
                meth=meth_test,
                cnv=cnv_test,
                clin=clin_test)
lapply(data.test, dim)

# make predictions with pre-trained model
predict.diablo=predict(res,
                       newdata=data.test,
                       dist='mahalanobis.dist')
predictions<-predict.diablo$WeightedVote$mahalanobis.dist[,ncomp]
print("made test set predictions")
confusion.mat = get.confusion_matrix(truth = Y_test, 
                                     predicted = predictions)                
# save predictions, confusion matrix
f<-snakemake@output[["testPred"]]
write.table(predictions, 
            file = f,
            quote = FALSE,
            sep = "\t",
            row.names = TRUE,
            col.names = FALSE) 
print("write test predictions")
f<-snakemake@output[["confusionMat"]]
write.table(confusion.mat, 
            file = f,
            quote = FALSE,
            sep = "\t",
            row.names = TRUE,
            col.names = TRUE) 
print("write confusion mat")

for (c in seq(1, ncomp))
{
# save balanced error rate, confusion mat for model w/t up to Nth component
predictions<-predict.diablo$WeightedVote$mahalanobis.dist[,c]
confusion.mat = get.confusion_matrix(truth = Y_test, 
                                     predicted = predictions)
f<-snakemake@log[[1]]
lapply(c(c, get.BER(confusion.mat)), 
         write,
         f, 
         append=TRUE)
}


