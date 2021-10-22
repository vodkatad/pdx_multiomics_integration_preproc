library(progeny)

# we need genes in rows and samples in columns
f <- snakemake@input[["expr_train"]]
train_expr <- data.matrix(read.csv(file = f, header=TRUE, row.names=1, sep="\t"))
f <- snakemake@input[["expr_test"]]
test_expr <- data.matrix(read.csv(file = f, header=TRUE, row.names=1, sep="\t"))


# calc PROGENy scores for test set
pathways <- progeny(test_expr)
write.csv(pathways, file = snakemake@output[["progeny_test"]], 
          row.names = TRUE, 
          quote=FALSE)

pathways <- progeny(train_expr)
write.csv(pathways, file = snakemake@output[["progeny_train"]], 
          row.names = TRUE, 
          quote=FALSE)