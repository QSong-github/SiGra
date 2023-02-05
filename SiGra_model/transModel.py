from torch_geometric.nn import TransformerConv, LayerNorm, GATConv, GCNConv
import torch.nn.functional as F
import torch
import math
# torch.set_default_tensor_type(torch.DoubleTensor)

class TranOne(torch.nn.Module):
    def __init__(self, hidden_dims, use_component='gene'):
        super().__init__()
        self.use_component = use_component

        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        # [in_dim, emb_dim, img_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = TransformerConv(in_dim, num_hidden)
        self.conv2 = TransformerConv(num_hidden, out_dim)
        self.conv3 = TransformerConv(out_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)

        self.imgconv1 = TransformerConv(img_dim, num_hidden)
        self.imgconv2 = TransformerConv(num_hidden, out_dim)
        self.imgconv3 = TransformerConv(out_dim, num_hidden)
        self.imgconv4 = TransformerConv(num_hidden, img_dim)

        # layernorm 
        self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)
        # relu
        self.activate = F.elu

    def forward(self, feat, edge):
        if self.use_component == 'gene':
            h1 = self.activate(self.conv1(feat, edge))
            h2 = self.conv2(h1, edge)
            h3 = self.activate(self.conv3(h2, edge))
            h4 = self.conv4(h3, edge)
        else:
            h1 = self.activate(self.imgconv1(feat, edge))
            h2 = self.imgconv2(feat, edge)
            h3 = self.activate(self.imgconv3(feat, edge))
            h4 = self.imgconv4(feat, edge)
        return h2, h4

class DataContrast(torch.nn.Module):
    def __init__(self, hidden_dims, ncluster, nspots):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GCNConv(in_dim, 2048)
        self.conv2 = GCNConv(2048, 4096)
        self.emb = GCNConv(4096, out_dim)

        self.conv3 = GCNConv(out_dim, num_hidden)
        self.conv4 = GCNConv(num_hidden, in_dim)

        mask = torch.Tensor(nspots, in_dim)
        torch.nn.init.uniform_(mask)
        mask = (mask > 0.5).float()
        self.mask = torch.nn.Parameter(mask)

        # self.mask = self.mask.float()

        # self.proj = TransformerConv(num_hidden, ncluster)
        self.proj = GCNConv(4096, ncluster)

        self.activate = F.elu
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, xi, edge_index):
        # print(xi.shape)
        hi1 = self.activate(self.conv1(xi, edge_index))
        hi2 = self.activate(self.conv2(hi1, edge_index))
        emb = self.activate(self.emb(hi2, edge_index))
        # combine1 = torch.concat([emb, hi2], dim=1)
        up1 = self.activate(self.conv3(emb, edge_index))
        # combine2 = torch.concat([up1, hi1], dim=1)
        up2 = self.conv4(up1, edge_index)
        ci = self.softmax(self.proj(hi2, edge_index))
        
        # print(xi.shape, self.mask.shape)
        xj = xi * self.mask
        hj1 = self.activate(self.conv1(xj, edge_index))
        hj2 = self.activate(self.conv2(hj1, edge_index))
        cj = self.softmax(self.proj(hj2, edge_index))

        return hi2, hj2, ci, cj, up2

class ImgContrast(torch.nn.Module):
    def __init__(self, hidden_dims, ncluster):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        self.imgconv1 = TransformerConv(img_dim, num_hidden)
        self.imgconv2 = TransformerConv(num_hidden, out_dim)

        self.proj = TransformerConv(out_dim, ncluster)
        self.activate = F.elu


    def forward(self, xi, xj, edge_index):
        hi1 = self.activate(self.imgconv1(xi, edge_index))
        hi2 = self.imgconv2(hi1, edge_index)
        ci = self.proj(hi1, edge_index)

        hj1 = self.activate(self.imgconv1(xj, edge_index))
        hj2 = self.imgconv(hj1, edge_index)
        cj = self.proj(hj2, edge_index)

        return hi2, hj2, ci, cj
    

class TransImg2(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        # [in_dim, emb_dim, img_dim, num_hidden, out_dim] = hidden_dims

        # self.conv1 = TransformerConv(in_dim, num_hidden)#, heads=1, dropout=0.1, beta=True)
        # # self.conv1 = TransformerConv(in_dim + emb_dim, num_hidden)#, heads=1, dropout=0.1, beta=True)
        # self.conv2 = TransformerConv(num_hidden, out_dim)#, heads=1, dropout=0.1, beta=True)
        # self.conv3 = TransformerConv(out_dim, num_hidden)#, heads=1, dropout=0.1, beta=True)
        # self.conv4 = TransformerConv(num_hidden, in_dim)#, heads=1, dropout=0.1, beta=True)

        self.imgconv1 = TransformerConv(img_dim, num_hidden)#, heads=1, dropout=0.1, beta=True)
        self.imgconv2 = TransformerConv(num_hidden, out_dim)#, heads=1, dropout=0.1, beta=True)
        self.imgconv3 = TransformerConv(out_dim, num_hidden)#, heads=1, dropout=0.1, beta=True)
        self.imgconv4 = TransformerConv(num_hidden, img_dim)#, heads=1, dropout=0.1, beta=True)

        # self.neck = TransformerConv(out_dim * 2, out_dim)#, heads=1, dropout=0.1, beta=True)
        # self.neck2 = TransformerConv(out_dim, out_dim)#, heads=1, dropout=0.1, beta=True)
        # self.c3 = TransformerConv(out_dim, num_hidden)#, heads=1, dropout=0.1, beta=True)
        # self.c4 = TransformerConv(num_hidden, in_dim)#, heads=1, dropout=0.1, beta=True)

        # layernorm 
        self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)
        # relu
        self.activate = F.elu

    def forward(self, features, img_feat, edge_index):
        # h1 = self.activate(self.conv1(features, edge_index))
        # h2 = self.conv2(h1, edge_index)
        # h3 = self.activate(self.conv3(h2, edge_index))
        # h4 = self.conv4(h3, edge_index)

        img1 = self.activate(self.imgconv1(img_feat, edge_index))
        img2 = self.imgconv2(img1, edge_index)
        img3 = self.activate(self.imgconv3(img2, edge_index))
        img4 = self.imgconv4(img3, edge_index)

        # concat = torch.cat([h2, img2], dim=1)
        # combine = self.activate(self.neck(concat, edge_index))
        # c2 = self.neck2(combine, edge_index)
        # c3 = self.activate(self.c3(c2, edge_index))
        # c4 = self.c4(c3, edge_index)

        # return h2, img2, c2, h4, img4, c4
        return img2, img4



# class TransImg(torch.nn.Module):
#     def __init__(self, hidden_dims, use_img_loss=False):
#         super().__init__()
#         [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
#         # [in_dim, emb_dim, img_dim, num_hidden, out_dim] = hidden_dims

#         self.conv1 = TransformerConv(in_dim, num_hidden)
#         self.conv2 = TransformerConv(num_hidden, out_dim)
#         self.conv3 = TransformerConv(out_dim, num_hidden)
#         self.conv4 = TransformerConv(num_hidden, in_dim)

#         self.imgconv1 = TransformerConv(img_dim, num_hidden)
#         self.imgconv2 = TransformerConv(num_hidden, out_dim)
#         self.imgconv3 = TransformerConv(out_dim, num_hidden)
#         if use_img_loss:
#             self.imgconv4 = TransformerConv(num_hidden, img_dim)
#         else:
#             self.imgconv4 = TransformerConv(num_hidden, in_dim)

#         self.neck = TransformerConv(out_dim * 2, out_dim)
#         self.neck2 = TransformerConv(out_dim, out_dim)
#         self.c3 = TransformerConv(out_dim, num_hidden)
#         self.c4 = TransformerConv(num_hidden, in_dim)

#         # layernorm 
#         self.norm1 = LayerNorm(num_hidden)
#         self.norm2 = LayerNorm(out_dim)
#         # relu
#         self.activate = F.elu

#     def forward(self, features, img_feat, edge_index):
#         h1 = self.activate(self.conv1(features, edge_index))
#         h2 = self.conv2(h1, edge_index)
#         h3 = self.activate(self.conv3(h2, edge_index))
#         h4 = self.conv4(h3, edge_index)

#         img1 = self.activate(self.imgconv1(img_feat, edge_index))
#         img2 = self.imgconv2(img1, edge_index)
#         img3 = self.activate(self.imgconv3(img2, edge_index))
#         img4 = self.imgconv4(img3, edge_index)

#         concat = torch.cat([h2, img2], dim=1)
#         combine = self.activate(self.neck(concat, edge_index))
#         c2 = self.neck2(combine, edge_index)
#         c3 = self.activate(self.c3(c2, edge_index))
#         c4 = self.c4(c3, edge_index)

#         # print(h4.shape, img4.shape, c4.shape)
#         return h2, img2, c2, h4, img4, c4

class TransImg(torch.nn.Module):
    def __init__(self, hidden_dims, use_img_loss=False):
        super().__init__()
        [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        # [in_dim, emb_dim, img_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = TransformerConv(in_dim, num_hidden)
        self.conv2 = TransformerConv(num_hidden, out_dim)
        self.conv3 = TransformerConv(out_dim, num_hidden)
        self.conv4 = TransformerConv(num_hidden, in_dim)

        self.imgconv1 = TransformerConv(img_dim, num_hidden)
        self.imgconv2 = TransformerConv(num_hidden, out_dim)
        self.imgconv3 = TransformerConv(out_dim, num_hidden)
        if use_img_loss:
            self.imgconv4 = TransformerConv(num_hidden, img_dim)
        else:
            self.imgconv4 = TransformerConv(num_hidden, in_dim)

        self.neck = TransformerConv(out_dim * 2, out_dim)
        self.neck2 = TransformerConv(out_dim, out_dim)
        self.c3 = TransformerConv(out_dim, num_hidden)
        self.c4 = TransformerConv(num_hidden, in_dim)

        # layernorm 
        self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)
        # relu
        self.activate = F.elu

    def forward(self, features, img_feat, edge_index):
        h1 = self.activate(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = self.activate(self.conv3(h2, edge_index))
        h4 = self.conv4(h3, edge_index)

        img1 = self.activate(self.imgconv1(img_feat, edge_index))
        img2 = self.imgconv2(img1, edge_index)
        img3 = self.activate(self.imgconv3(img2, edge_index))
        img4 = self.imgconv4(img3, edge_index)

        concat = torch.cat([h2, img2], dim=1)
        combine = self.activate(self.neck(concat, edge_index))
        c2 = self.neck2(combine, edge_index)
        c3 = self.activate(self.c3(c2, edge_index))
        c4 = self.c4(c3, edge_index)

        # print(h4.shape, img4.shape, c4.shape)
        return h2, img2, c2, h4, img4, c4
