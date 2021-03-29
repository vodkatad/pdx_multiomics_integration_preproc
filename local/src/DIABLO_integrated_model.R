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
minfeatures<-snakemake@params[["minfeatures"]]
maxfeatures<-snakemake@params[["maxfeatures"]]
step<-snakemake@params[["step"]]
nfeatures_seq<-c(seq(minfeatures, maxfeatures, step))
test.keepX=list("expr"=nfeatures_seq,
                "mut"=c(seq(minfeatures, ncol(mut), step)),
               "meth"=nfeatures_seq,
                "cnv"=nfeatures_seq)
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



