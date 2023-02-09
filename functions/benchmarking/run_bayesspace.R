args = commandArgs(trailingOnly=TRUE)
dataroot <- args[1]
num_fovs <- args[2]
n_clusters <- as.integer(args[3])
savepath <- args[4]
sample.name <- args[5]


library(BayesSpace)
library(ggplot2)
library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(anndata)
library(dplyr)


show_BayesSpace <- function(input_root, num_fovs, n_cluster, savepath, sample.name){
    for (i in 1:num_fovs){
        fovid = paste0('fov',i)
        input = file.path(input_root, fovid, 'sampledata.h5seurat')
        output = file.path(savepath, sample.name)
        if (!dir.exists(file.path(output))){
            dir.create(file.path(output), recursive = TRUE)
        }

        dlpfc = LoadH5Seurat(input, meta.data = FALSE)
        ab <- colSums(dlpfc)

        print(input)
        library(rhdf5)
        foo <- h5read(file.path(input_root, fovid,'sampledata.h5ad'), "obs")
        foo1 = data.frame(
                  row= foo$cx, cx_g = foo$cx, col = foo$cy,
                  cy_g = foo$cy, 
                  merge_cell_code = foo$merge_cell_type$codes, sums=ab)
        dlpfc@meta.data = foo1
        dlpfc <- subset(dlpfc,sums!=0)
        foo1 <- dlpfc@meta.data
        X <- dlpfc[['RNA']]@counts
        # X <- dlpfc[['X']]@counts

        dlpfc <- SingleCellExperiment(X,colData = dlpfc@meta.data)
        libsizes <- colSums(X)
        size.factors <- libsizes/mean(libsizes)
        logcounts(dlpfc) <- log2(t(t(X)/size.factors) + 1)

        set.seed(1234)
        dec <- scran::modelGeneVar(dlpfc)
        top <- scran::getTopHVGs(dec, n = 500)

        set.seed(1234)
        dlpfc <- scater::runPCA(dlpfc, subset_row=top)

        ## Add BayesSpace metadata
        dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)
        
        ##### Clustering with BayesSpace
        q <- n_clusters  # Number of clusters
        d <- 15  # Number of PCs

        ## Run BayesSpace clustering
        set.seed(1234)
        dlpfc <- spatialCluster(dlpfc, q=q, d=d, nrep=5000, gamma=3, platform="Visium", save.chain=FALSE)
        labels <- dlpfc$spatial.cluster
        label <- data.frame(pred=labels,gt=foo1$merge_cell_code)
        gp <- dlpfc$layer_guess

        write.csv(label,file=file.path(output,paste(fovid,'_bayesSpace.csv',sep='')), quote=F)
        write.table(colData(dlpfc),file=file.path(output,paste(fovid,'_bayesSpace_emb.csv', sep='')), quote=FALSE)
    }
}

show_BayesSpace(dataroot, num_fovs, n_clusters, savepath, sample.name)