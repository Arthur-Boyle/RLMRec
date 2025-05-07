import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add


from config.configurator import configs
from models.general_cf.lightgcn import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DCCF(BaseModel):
    def __init__(self, data_handler):
        super(DCCF, self).__init__(data_handler)

        # prepare adjacency matrix for DCCF
        rows = data_handler.trn_mat.tocoo().row
        cols = data_handler.trn_mat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.user_num + self.item_num, self.user_num + self.item_num]).tocsr().tocoo()
        # 强制转换 plain_adj.row 与 plain_adj.col 为 np.int64
        rows_int64 = plain_adj.row.astype(np.int64)
        cols_int64 = plain_adj.col.astype(np.int64)

        self.A_in_shape = plain_adj.shape

    # 直接从 NumPy 数组转换到 Tensor，并指定数据类型为 torch.long（int64）
        self.all_h_list = torch.tensor(rows_int64, dtype=torch.long, device='cuda')
        self.all_t_list = torch.tensor(cols_int64, dtype=torch.long, device='cuda')

        

    # 用 torch.stack 拼接索引
        self.A_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0).cuda()

        self.D_indices = torch.tensor(
            [list(range(self.user_num + self.item_num)), list(range(self.user_num + self.item_num))],
            dtype=torch.long, device='cuda'
        )

        self.G_indices, self.G_values = self._cal_sparse_adj()

        # hyper-parameter
        self.intent_num = configs['model']['intent_num']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.temperature = self.hyper_config['temperature']

        # model parameters
        self.user_embeds = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_size)
        self.user_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = torch.nn.Parameter(init(torch.empty(self.embedding_size, self.intent_num)), requires_grad=True)

        # train/test
        self.is_training = True
        self.final_embeds = False

        self._init_weight()
        self.gat_layer = GATLayer(in_dim=self.embedding_size, out_dim=self.embedding_size, num_heads=1)

    def _init_weight(self):
        init(self.user_embeds.weight)
        init(self.item_embeds.weight)

    def _cal_sparse_adj(self):
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        print("A_in_shape =", self.A_in_shape)
        print("self.all_h_list.max() =", self.all_h_list.max().item())
        print("self.all_t_list.max() =", self.all_t_list.max().item())
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape,trust_data=True).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha, sparse_sizes=self.A_in_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return G_indices, G_values

    def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None, None

        all_embeds = [torch.concat([self.user_embeds.weight, self.item_embeds.weight], dim=0)]
        gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = [], [], [], []

        for i in range(0, self.layer_num):
            # 使用 GAT 层进行图注意力聚合
            gnn_layer_embeds = self.gat_layer(all_embeds[i], self.G_indices)  # 使用图的邻接矩阵 G_indices

            # Intent-aware Information Aggregation
            u_embeds, i_embeds = torch.split(all_embeds[i], [self.user_num, self.item_num], 0)
            u_int_embeds = torch.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = torch.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = torch.concat([u_int_embeds, i_int_embeds], dim=0)

            # Adaptive Augmentation
            gnn_head_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_h_list)
            gnn_tail_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_t_list)
            int_head_embeds = torch.index_select(int_layer_embeds, 0, self.all_h_list)
            int_tail_embeds = torch.index_select(int_layer_embeds, 0, self.all_t_list)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeds, gnn_tail_embeds)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeds, int_tail_embeds)
            gaa_layer_embeds = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])
            iaa_layer_embeds = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Aggregation
            gnn_embeds.append(gnn_layer_embeds)
            int_embeds.append(int_layer_embeds)
            gaa_embeds.append(gaa_layer_embeds)
            iaa_embeds.append(iaa_layer_embeds)
            all_embeds.append(gnn_layer_embeds + int_layer_embeds + gaa_layer_embeds + iaa_layer_embeds + all_embeds[i])

        all_embeds = torch.stack(all_embeds, dim=1)
        all_embeds = torch.sum(all_embeds, dim=1, keepdim=False)
        user_embeds, item_embeds = torch.split(all_embeds, [self.user_num, self.item_num], 0)
        self.final_embeds = all_embeds
        return user_embeds, item_embeds, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds

    def _cal_cl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items) # different from original SSLRec, remove negative items
        cl_loss = 0.0
        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.user_num, self.item_num], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.user_num, self.item_num], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.user_num, self.item_num], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.user_num, self.item_num], 0)

            u_gnn_embs = u_gnn_embs[users]
            u_int_embs = u_int_embs[users]
            u_gaa_embs = u_gaa_embs[users]
            u_iaa_embs = u_iaa_embs[users]

            i_gnn_embs = i_gnn_embs[items]
            i_int_embs = i_int_embs[items]
            i_gaa_embs = i_gaa_embs[items]
            i_iaa_embs = i_iaa_embs[items]

            cl_loss += cal_infonce_loss(u_gnn_embs, u_int_embs, u_int_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_gaa_embs, u_gaa_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_iaa_embs, u_iaa_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_int_embs, i_int_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_gaa_embs, i_gaa_embs, self.temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_iaa_embs, i_iaa_embs, self.temperature) / u_gnn_embs.shape[0]
        return cl_loss

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds)
        loss = bpr_loss + reg_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _, _, _ = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim * num_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        # 注意：这里 self.a 的形状为 (2*out_dim, num_heads)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, num_heads)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, h, adj_indices):
        """
        h: 节点嵌入 (num_nodes, in_dim)
        adj_indices: 邻接矩阵的边索引 (2, num_edges)
        """
        h_prime = torch.matmul(h, self.W)
        h_prime = h_prime.view(-1, self.num_heads, self.out_dim)
        num_edges = adj_indices.shape[1]

        head_outputs = []
        for head in range(self.num_heads):
            h_prime_head = h_prime[:, head, :]  # (num_nodes, out_dim)

            src_nodes = adj_indices[0]  # (num_edges,)
            dst_nodes = adj_indices[1]  # (num_edges,)
            h_src = h_prime_head[src_nodes]  # (num_edges, out_dim)
            h_dst = h_prime_head[dst_nodes]  # (num_edges, out_dim)

            edge_features = torch.cat([h_src, h_dst], dim=-1)  # (num_edges, 2*out_dim)

            att = (edge_features * self.a[:, head].unsqueeze(0)).sum(dim=-1)
            att = F.leaky_relu(att, negative_slope=self.alpha)

            # ✅ 正确的 attention softmax（每个目标节点的入边上归一化）
            att = scatter_softmax(att, dst_nodes)

            # attention 加权源节点特征
            aggregated = scatter_add(att.unsqueeze(-1) * h_src, dst_nodes, dim=0, dim_size=h_prime_head.size(0))

            head_outputs.append(aggregated)

        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout_layer(out)
        return out



