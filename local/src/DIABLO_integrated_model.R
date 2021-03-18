# see: https://github.com/NikolayOskolkov/SupervisedOMICsIntegration/blob/master/supervised_omics_integr_CLL.Rmd
library("mixOmics")
require(data.table)

# load train data for e/a omic
f <- snakemake@input[["expr_Xtrain"]]
expr <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(expr)<-expr$V1
expr<-subset(expr, select = -c(V1))
print("loaded expr")
f <- snakemake@input[["meth_Xtrain"]]
meth <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(meth)<-rownames(expr)
meth<-subset(meth, select = -c(V1))
print("loaded meth")
f <- snakemake@input[["cnv_Xtrain"]]
cnv <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(cnv)<-rownames(expr)
cnv<-subset(cnv, select = -c(V1))
print("loaded cnv")
f <- snakemake@input[["mut_Xtrain"]]
mut <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(mut)<-rownames(expr)
mut<-subset(mut, select = -c(V1))
print("loaded mut")
# load target
f <- snakemake@input[["Ytrain"]]
Y<-as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(Y)<-rownames(expr)
target_col<-quote(snakemake@params[["target_col"]])
Y<-factor(Y[,eval(target_col)])
print("loaded Y")

# integrate the omics
data<-list(expr=expr, mut=mut, meth=meth, cnv=cnv)
print("listed data")
#first fit a DIABLO model without variable selection to assess 
# the global performance and choose the number of components for the final model. 

# the design matrix (omics x omics) sets the expected covariance between the OMICs chosen based on prior knowledge.
# Here due to the lack of prior knowledge we select a strong correlation 1 between the OMICs.
design=matrix(1,ncol=length(data),
              nrow=length(data),
              dimnames=list(names(data),names(data)))
diag(design)=0
print("build design matrix")
# assess the performance of this first model
# perf is run with nfold-fold cross validation for nrepeat times. 
splsda.res = block.splsda(X = data, Y = Y, ncomp = 8, design = design)
print("set model")
perf.diablo = perf(splsda.res,
                   validation = 'Mfold', 
                   folds = 2, 
                   nrepeat = 5, 
                   progressBar=FALSE,
                   cpus=snakemake@threads)
print("check performance")
# plot error rate by number of components used by the model
f <- snakemake@output[["ncompBER_plot"]]
pdf(f,
    width = 8,
    height = 5,
    bg = "white", 
    colormodel = "cmyk",
    paper = "A4") 
plot(perf.diablo) 
dev.off() 
print("plot ncompBER")

# tune the model on the train set
# optimise the number of features for each 'omic, component
test.keepX=list("expr"=c(seq(5, 30, 10)),
                "mut"=c(seq(5, ncol(mut), 5)),
               "meth"=c(seq(5, 30, 10)),
                "cnv"=c(seq(5, 30, 10)))
max_ncomp<-snakemake@params[["max_ncomp"]]
tune.omics=tune.block.splsda(X=data,
                             Y=Y,
                             ncomp=max_ncomp,
                             test.keepX=test.keepX,
                             design=design,
                             cpus=snakemake@threads,
                             progressBar=FALSE,
                             validation="Mfold",
                             folds=4,
                             nrepeat=50,
                             near.zero.var=FALSE,
                             dist = "mahalanobis.dist")
print("tuned the model")
# train final sPLS-DA model,
# use optimal number of components, features
optimal_ncomp<-tune.omics$choice.ncomp$ncomp 
list.keepX=list("expr"=first(tune.omics$choice.keepX$expr, optimal_ncomp),
                "mut"=first(tune.omics$choice.keepX$mut, optimal_ncomp),
                "meth"=first(tune.omics$choice.keepX$meth, optimal_ncomp),
                "cnv"=first(tune.omics$choice.keepX$cnv, optimal_ncomp))
res=block.splsda(X=data,
                 Y=Y,
                 ncomp=optimal_ncomp,
                 keepX=list.keepX,
                 design=design,
                 near.zero.var=FALSE)
print("built final model")

# save trained model as RDS file
f <- snakemake@output[["splsda_model"]] 
saveRDS(res, file = f)
print("saved model")

# assess performance on train set (4-fold CV x 100 repeats) 
# with Weighted prediction
perf.res <- perf(res, validation = "Mfold", 
                 folds = 4,  nrepeat = 100,
                 dist = "mahalanobis.dist") 
print("check model perf on training set")

# log balanced error rate on training set
perf_res_train<-as.data.frame(perf.res$WeightedVote.error.rate)
f <- snakemake@output[["trainBER_table"]]
write.table(perf_res_train, 
            file = f,
            quote = FALSE, 
            sep = "\t",
            row.names = TRUE,
            col.names = TRUE)
print("save train BER")

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
print("plot indiv")

# show and save loadings for each omic and each component
f<-snakemake@output[["PlotLoadings"]]
pdf(f,
    width = 10, height = 6, bg = "white", 
    colormodel = "cmyk",paper = "A4")
for (c in seq(0, optimal_ncomp))
{
    plotLoadings(res,comp=c,contrib='max',method='median')
}
dev.off()
print("plot loadings")
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
print("write loadings")

# Now we will diplay each individual on the plot of each of the OMICs
# against each other which are defined by the top loadings from 
# their respective components.
f<-snakemake@output[["plotDiablo"]]
pdf(f,
    width = 10, height = 6, bg = "white", 
    colormodel = "cmyk",paper = "A4")
for (c in seq(0, optimal_ncomp))
{
   plotDiablo(res,ncomp=c) 
}
dev.off()
print("plot Diablo")

# Now let us display so-called “arrow plot”  
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
print("plot arrow")

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
# of gene expression, methylation, mutation, adn CNV 
# that provide clustering of individuals into females and males.
f<-snakemake@output[["corrHeatmap"]]
pdf(f,
    width = 24, height = 14, bg = "white", 
    colormodel = "cmyk",paper = "A4") 
cimDiablo(res)
dev.off()
print("plot heatmap")

### Predict target classes from OMICs Integration,
# assess prediction accuracy

# load test data for e/a omic
f <- snakemake@input[["expr_Xtest"]]
expr_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(expr_test)<-expr_test$V1
expr_test<-subset(expr_test, select = -c(V1))
f <- snakemake@input[["meth_Xtest"]]
meth_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(meth_test)<-rownames(expr_test)
meth_test<-subset(meth_test, select = -c(V1))
f <- snakemake@input[["cnv_Xtest"]]
cnv_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(cnv_test)<-rownames(expr_test)
cnv_test<-subset(cnv_test, select = -c(V1))
f <- snakemake@input[["mut_Xtest"]] 
mut_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(mut_test)<-rownames(expr_test)
mut_test<-subset(mut_test, select = -c(V1))
# load target
f <- snakemake@input[["Ytest"]] 
Y_test<-as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(Y_test)<-rownames(expr_test)
Y_test<-subset(Y_test, select = -c(V1))
Y_test<-factor(Y_test[, eval(target_col)])
print("load testfiles")

# build test set
data.test<-list(expr=expr_test,mut=mut_test,meth=meth_test,cnv=cnv_test)
lapply(data.test, dim)

# make predictions
predict.diablo=predict(res,
                       newdata=data.test,
                       dist='mahalanobis.dist')
print("make predictions")

predictions<-predict.diablo$WeightedVote$mahalanobis.dist[,optimal_ncomp]
confusion.mat = get.confusion_matrix(truth = Y_test, 
                                     predicted = predictions)
# save confusion matrix
f<-snakemake@output[["confusionMat"]]
write.table(confusion.mat, 
            file = f,
            quote = FALSE,
            sep = "\t",
            row.names = TRUE,
            col.names = TRUE) 
print("write confusion mat")

for (c in seq(0, optimal_ncomp))
{
# save balanced error rate for e/a component
f<-snakemake@log[[1]]
lapply(c(c, get.BER(confusion.mat)), 
         write,
         f, 
         append=TRUE)
}


