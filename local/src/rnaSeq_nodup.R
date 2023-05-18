inputMetadata <- snakemake@input[['metadata']]
inputExpr <- snakemake@input[['rnaSeq_expr']]
output <- snakemake@output[['rnaSeq']]

# Load the selected metadata table
metaData <- read.table(inputMetadata, sep="\t", header=TRUE)

# Keep the cases with LMX_BASALE type
lmxBas <- grepl("LMX_BASALE", metaData$type)
md_lmxBas <- metaData[lmxBas,]

# Load the selected RNA seq table
rna_seq <- read.table(inputExpr, sep="\t", header=TRUE)

# Extract the sample IDs of the md_lmxBas
col_names <- as.vector(md_lmxBas$sample_id_R)

# Extract from the RNA seq dataset the samples with the LMX_BASALE type 
rnaSeq_lmxBas <- rna_seq[,col_names]

# Extract the first 7 characters of the sample IDs
seven_chars <- substr(colnames(rnaSeq_lmxBas), 1, 7)

# Find the duplicates in the seven_chars1 
dup <- seven_chars[duplicated(seven_chars)]
no_dup <- seven_chars[!duplicated(seven_chars)]

numb_genes <- nrow(rna_seq)
numb_dup <- length(dup)
numb_nodup <- length(no_dup)

# Create a data frame without the duplicates
rnaSeq_lmxBas_noDup <- data.frame(matrix(NA, nrow=numb_genes))

for(i in seq(1, numb_nodup)){
  d <- grepl(no_dup[i], colnames(rnaSeq_lmxBas))
  rnaSeq_lmxBas_noDup <- cbind(rnaSeq_lmxBas[, d, drop=FALSE], rnaSeq_lmxBas_noDup)
}


# Create a data frame with the duplicates only
rnaSeq_lmxBas_dup <- data.frame(matrix(NA, nrow=numb_genes))

for(i in seq(1, numb_dup)){
  d <- grepl(dup[i], colnames(rnaSeq_lmxBas))
  rnaSeq_lmxBas_dup <- cbind(rnaSeq_lmxBas[, d], rnaSeq_lmxBas_dup)
}

# Calculating the mean of expression of all genes for duplicates

rnaSeq_mean <- data.frame(matrix(NA, nrow = numb_genes, ncol = numb_dup))
colnames(rnaSeq_mean) <- dup

for(i in seq(1, length(dup))){
  e <- data.frame(matrix(NA, nrow=numb_genes))
  d <- grepl(dup[i], colnames(rnaSeq_lmxBas_dup))
  e <- cbind(rnaSeq_lmxBas_dup[, d], e)
  e$RowMeans<-rowMeans(e,na.rm=TRUE)
  rnaSeq_mean[,dup[i]] <- e[,"RowMeans"]
}

rnaSeq_mean <- cbind(rna_seq$Geneid, rnaSeq_mean)

# Rename rna_seq$Geneid column to Geneid
names(rnaSeq_mean)[names(rnaSeq_mean) == "rna_seq$Geneid"] <- "Geneid"

# Combine the rnaSeq_mean and rnaSeq_lmxBas_noDup data frames 
rnaSeq <- cbind(rnaSeq_mean, rnaSeq_lmxBas_noDup)
  
#save the rnaSeq data frame as a tsv file
write.table(rnaSeq, file=output, quote=FALSE, sep='\t')