library(progeny)

# we need genes in rows and samples in columns
gene_expr <- data.matrix(read.csv(file = "tables/CMP+GDSCexpr_X_all_T.tsv", 
                                    header=TRUE, row.names=1, sep="\t"))
pathways <- progeny(gene_expr)

write.csv(pathways, file = "tables/progenyPath_CMP+GDSCexpr_X_all.tsv", 
          row.names = TRUE, quote=FALSE)