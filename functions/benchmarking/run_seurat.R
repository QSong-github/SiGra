args = commandArgs(trailingOnly=TRUE)
dataroot <- args[1]
num_fovs <- args[2]
n_clusters <- args[3]
savepath <- args[4]
sample.name <- args[5]

# sample.name <- paste0(sample,'/',fov)

#' sample='nano_5-1'
#' fov='fov1'

library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(ggplot2)
library(patchwork)
library(dplyr)
options(bitmapType = 'cairo')
library(anndata)

# dir.input <- file.path('../dataset/', sample.name)
# dir.output <- file.path('./seurat/', sample.name)

# create h5seurat first
# convert_h5seurat <- function(input_root, num_fovs){
#   for (i in 1:num_fovs){
#     fovid = paste0('fov',i)
#     input = file.path(input_root, fovid, 'sampledata.h5ad')
#     print(input)
#     Convert(input, dest = "h5seurat", overwrite = T)
#   }
# }
# convert_h5seurat(dataroot, num_fovs)

# run seurat
show_seurat <- function(input_root, num_fovs, n_cluster, savepath, sample.name){
    for (i in 1:num_fovs){
        fovid = paste0('fov',i)
        print(fovid)
        input = file.path(input_root, fovid, 'sampledata.h5seurat')
        output = file.path(savepath, sample.name)
        if (!dir.exists(file.path(output))){
            dir.create(file.path(output), recursive = TRUE)
        }

        sp_data = LoadH5Seurat(input, meta.data = FALSE)
        # remove nan from sp_data
        sp_data = na.omit(sp_data)
        ab <- colSums(sp_data)
        sp_data@meta.data$sums <- ab
        sp_data <- subset(sp_data,sums!=0)
        # sp_data = subset(sp_data, subset = nCount_Spatial > 0)
        print('sctransofrom')
        set.seed(1234)
        sp_data <- SCTransform(sp_data, assay = "RNA", verbose = T, variable.features.n = 980)
        set.seed(1234)
        sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)
        sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:50)
        for(resolution in 5:50){
            set.seed(1234)
            sp_data <- FindClusters(sp_data, verbose = F, resolution = resolution/100)
            # print(length(levels(sp_data@meta.data$seurat_clusters)))
            if(length(levels(sp_data@meta.data$seurat_clusters)) == n_clusters){
                break
            }
        }
        set.seed(1234)
        sp_data <- FindClusters(sp_data, verbose = FALSE, resolution = resolution/100)
        # print(length(levels(sp_data@meta.data$seurat_clusters)))

        saveRDS(sp_data, file.path(output, paste(fovid,'_Seurat_final.rds',sep='')))
        write.table(sp_data@reductions$pca@cell.embeddings, file = file.path(output, paste(fovid,'_seurat.PCs.csv',sep='')), sep='\t', quote=F)
        write.table(sp_data@meta.data, file = file.path(output, paste(fovid,'.csv',sep='')), sep='\t', quote=FALSE)
    }
}
show_seurat(dataroot, num_fovs, n_clusters, savepath, sample.name)