# see: https://github.com/NikolayOskolkov/SupervisedOMICsIntegration/blob/master/supervised_omics_integr_CLL.Rmd
library("mixOmics")
require(data.table)

# load train data for e/a omic
f <- snakemake@input[["expr_Xtrain"]]
expr <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(expr)<-expr$V1
expr<-subset(expr, select = -c(V1))
f <- "tables/PDx_Meth_MultiFeatSelect_XtrainClean.tsv"
f <- snakemake@input[["meth_Xtrain"]]
meth <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(meth)<-rownames(expr)
meth<-subset(meth, select = -c(V1))
f <- snakemake@input[["cnv_Xtrain"]]
cnv <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(cnv)<-rownames(expr)
cnv<-subset(cnv, select = -c(V1))
f <- snakemake@input[["mut_Xtrain"]]
mut <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(mut)<-rownames(expr)
mut<-subset(mut, select = -c(V1))

# load target
f <- snakemake@input[["Ytrain"]]
Y<-as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(Y)<-rownames(expr)
Y<-factor(Y$snakemake@params[["target_col"]])

# integrate the omics
data<-list(expr=expr, mut=mut, meth=meth, cnv=cnv)

#first fit a DIABLO model without variable selection to assess 
# the global performance and choose the number of components for the final model. 

# the design matrix (omics x omics) sets the expected covariance between the OMICs chosen based on prior knowledge.
# Here due to the lack of prior knowledge we select a strong correlation 1 between the OMICs.
design=matrix(1,ncol=length(data),
              nrow=length(data),
              dimnames=list(names(data),names(data)))
diag(design)=0

# assess the performance of this first model
# perf is run with nfold-fold cross validation for nrepeat times. 
splsda.res = block.splsda(X = data, Y = Y, ncomp = 8, design = design)
perf.diablo = perf(splsda.res, validation = 'Mfold', 
                   folds = 2, nrepeat = 5, 
                   progressBar=FALSE, cpus=4)
# plot error rate by number of components used by the model
pdf(snakemake@output[["ncompBER_plot"]]
    width = 8, height = 5, bg = "white", 
    colormodel = "cmyk",paper = "A4") 
plot(perf.diablo) 
dev.off() 

# optimise the number of features for each 'omic, component
test.keepX=list("expr"=c(seq(5, 30, 5)),
                "mut"=c(seq(5, 15, 5)),
               "meth"=c(seq(5, 30, 5)),
                "cnv"=c(seq(5, 30, 5)))

#optimal_ncomp<-perf.diablo$choice.ncomp$WeightedVote["Overall.BER", "mahalanobis.dist"]
optimal_ncomp<-3
tune.omics=tune.block.splsda(X=data,Y=Y,
                             ncomp=optimal_ncomp,test.keepX=test.keepX,
                             design=design,cpus=4,progressBar=TRUE,
                             validation="Mfold",folds=4,nrepeat=10,
                             near.zero.var=FALSE,
                             dist = "mahalanobis.dist")

# final sPLS-DA modelling, display PCA plots and loadings. 
optimal_ncomp<-tune.omics$choice.ncomp$ncomp
list.keepX=list("expr"=first(tune.omics$choice.keepX$expr, optimal_ncomp),
                "mut"=first(tune.omics$choice.keepX$mut, optimal_ncomp),
                "meth"=first(tune.omics$choice.keepX$meth, optimal_ncomp),
                "cnv"=first(tune.omics$choice.keepX$cnv, optimal_ncomp))

res=block.splsda(X=data,Y=Y,
                 ncomp=optimal_ncomp,
                 keepX=list.keepX,
                 design=design,
                 near.zero.var=FALSE)

saveRDS(res, file = "models/DIABLOsplda.rds")
perf.res <- perf(res, validation = "Mfold", 
                 folds = 4,  nrepeat = 100,
                 dist = "mahalanobis.dist") 

# Performance on train set (2-fold CV x 100 repeats) with Weighted prediction
perf.res$WeightedVote.error.rate
perf_res_train<-as.data.frame(perf.res$WeightedVote.error.rate)
write.table(perf_res_train, file = "tables/DIABLOsplda_train_BER.tsv",
            quote = FALSE, sep = " \t",
            row.names = TRUE,  col.names = TRUE)

# Low-dimensional latent PLS space representation of each individual OMIC
# samples are represented as points placed according to 
# their projection in the smaller subspace spanned by the components
pdf("figures/DIABLO_sPLSDA_plotIndiv.pdf",
    width = 10, height = 6, bg = "white", 
    colormodel = "cmyk",paper = "A4") 
plotIndiv(res,legend=TRUE,
          title="PDx Omics sPLS-DA",
          ellipse=FALSE,
          ind.names=FALSE,cex=.5,
          #style = 'graphics'
          )
dev.off() 

# show loadings for each omic and each component
plotLoadings(res,comp=1,contrib='max',method='median')
plotLoadings(res,comp=2,contrib='max',method='median')
plotLoadings(res,comp=3,contrib='max',method='median')

# save mut loadings separately
res_mut_loadings<-as.data.frame(plotLoadings(res, comp=2, 
                                             contrib='max',
                                             method='median', 
                                             plot=FALSE, 
                                             block = "mut"))
write.table(res_mut_loadings, file = "tables/DIABLOsplda_train_mutLoadings.tsv",
            quote = FALSE, sep = " \t",
            row.names = TRUE,  col.names = TRUE)


# Now we will diplay each individual on the plot of each of the OMICs
# against each other which are defined by the top loadings from their respective components.
plotDiablo(res,ncomp=1)
plotDiablo(res,ncomp=2)

# Now let us display so-called “arrow plot” which demonstrates the samples (individuals) 
# in a superimposed manner where each sample will be indicated using an arrow. 
# The start of the arrow indicates the location of the sample in X in one plot, 
# and the tip the location of the sample in Y in the other plot. 
# Short arrows indicate a strong agreement between the matching data sets, 
# long arrows a disagreement between the matching data sets.
# Such graphic highlight the agreement between all data sets at the sample level, when modelled with DIABLO.
pdf("figures/DIABLO_sPLSDA_plotArrow.pdf",
    width = 8, height = 6, bg = "white",
    colormodel = "cmyk",paper = "A4") 
plotArrow(res,ind.names=FALSE,legend=TRUE,
          title="PDx omics integration arrow plot")
dev.off() 

# calculate “circos plot” that diaplays variable 
# correlation among different OMICs datasets
# the variables for this plot were selected simultaneously from all the OMICs, i.e. they are 
# not equavalent to those obtained from each individual OMIC separately.
pdf("figures/DIABLO_sPLSDA_circosPlot.pdf",
    width = 18, height = 16, bg = "white", 
    colormodel = "cmyk",paper = "A4") 
circosPlot(res,cutoff=.5,line=FALSE,size.variables=0.5)
dev.off()

# Correlation network is another way to demostrate correlations 
# between top loadings of the OMICs data sets in a pairwise fashion.
network(res,blocks=c(1,2),cex.node.name=0.6,color.node=c('blue','red2'),breaks=NULL)


# Finally the correlation heatmap displays strongly correlated blocks of gene expression, 
# methylation and clinical variables markers that provide clustering of individuals into females and males.
pdf("figures/DIABLO_sPLSDA_corrHetMap.pdf",
    width = 24, height = 14, bg = "white", 
    colormodel = "cmyk",paper = "A4") 
cimDiablo(res)
dev.off()

### Predict Cetuximab 3w response (cat) from OMICs Integration,
# assess the prediction accuracy

# load test data for e/a omic
f <- "tables/PDx_Expr_MultiFeatSelect_XtestClean.tsv"
expr_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(expr_test)<-expr_test$V1
expr_test<-subset(expr_test, select = -c(V1))
f <- "tables/PDx_Meth_MultiFeatSelect_XtestClean.tsv"
meth_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(meth_test)<-rownames(expr_test)
meth_test<-subset(meth_test, select = -c(V1))
f <- "tables/PDx_CNV_FeatSelect_XtestClean.tsv"
cnv_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(cnv_test)<-rownames(expr_test)
cnv_test<-subset(cnv_test, select = -c(V1))
f <- "tables/PDx_driverMutVAF_FeatSelect_XtestClean.tsv"
mut_test <- as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(mut_test)<-rownames(expr_test)
mut_test<-subset(mut_test, select = -c(V1))
# load target
f <- "tables/Cetuximab_3w_cat_testClean.tsv"
Y_test<-as.data.frame(fread(f, header = TRUE, sep="\t"))
rownames(Y_test)<-rownames(expr_test)
Y_test<-subset(Y_test, select = -c(V1))
Y_test<-factor(Y_test$Cetuximab_Standard_3wks_cat)

# build test design matrix
data.test<-list(expr=expr_test,mut=mut_test,meth=meth_test,cnv=cnv_test)
lapply(data.test, dim)

predict.diablo=predict(res,newdata=data.test,dist='mahalanobis.dist')
predictions<-predict.diablo$WeightedVote$mahalanobis.dist[,1]
confusion.mat = get.confusion_matrix(truth = Y_test, 
                                     predicted = predictions)
get.BER(confusion.mat)
confusion.mat
