import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from config.configurator import configs
from models.general_cf.lightgcn import BaseModel
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DCCF_LK(BaseModel):
    def __init__(self, data_handler):
        super(DCCF_LK, self).__init__(data_handler)

        # 使用LightGCN中的邻接矩阵
        self.adj = data_handler.torch_adj

        self.latent_dim = self.hyper_config['embedding_size']       # 嵌入向量维度
        self.k = self.hyper_config['num_clusters']                         # 聚类的类别数（k-means 中的 k）
        self.ssl_temp = self.hyper_config['ssl_temp']           # ProtoNCE 温度参数
        self.proto_reg = float(self.hyper_config['proto_reg'])  # ProtoNCE 正则项权重
        self.device = "cuda"

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

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

    def _init_weight(self):
        init(self.user_embeds.weight)
        init(self.item_embeds.weight)

    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)
    
    def e_step(self):
        """
        E-step：利用当前的用户和物品嵌入，通过 K-means 得到对应的原型（centroid）和簇分配
        """
        # 这里假设 self.user_embedding 和 self.item_embedding 已初始化且权重与 self.user_embeds 保持一致
        user_embeddings = self.user_embeds.weight.detach().cpu().numpy()
        item_embeddings = self.item_embeds.weight.detach().cpu().numpy()


        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """
        Run K-means algorithm to get k clusters of the input tensor x
        """
        import faiss
        # 使用 faiss 的 GPU k-means
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)

        # 转换为 cuda Tensor 以便后续广播操作
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None, None

        all_embeds = [torch.concat([self.user_embeds.weight, self.item_embeds.weight], dim=0)]
        gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = [], [], [], []

        for i in range(0, self.layer_num):
            # 使用 LightGCN 的传播方式进行图卷积
            gnn_layer_embeds = self._propagate(self.adj, all_embeds[i])

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
        items = torch.unique(items)  # different from original SSLRec, remove negative items
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
        cecenter_embedding = torch.concat([user_embeds, item_embeds], dim=0)
        ancs, poss, negs = batch_data
        # 聚类自监督的 ProtoNCE 损失
        proto_nce_loss = self.ProtoNCE_loss(cecenter_embedding, 
                                            ancs,  # 假定batch_data中包含用户下标
                                            poss)  # 以及物品下标，根据实际情况调整
        loss = bpr_loss + reg_loss + proto_nce_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}

        #去掉cl损失
        #loss = bpr_loss + reg_loss + proto_nce_loss 
        #losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha, sparse_sizes=self.A_in_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return G_indices, G_values

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _, _, _, _ = self.forward()
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds

    def ProtoNCE_loss(self, node_embedding, user, item):
        """
        计算基于原型的对比损失（ProtoNCE Loss）
        参数：
            node_embedding: 拼接的用户和物品嵌入 [n_users+n_items, e]
            user: 采样的用户下标张量
            item: 采样的物品下标张量
        """
        # 确保 user 和 item 是 LongTensor 且没有多余的维度
        user = user.squeeze().long()
        item = item.squeeze().long()

        # 拆分出用户和物品整体嵌入
        user_embeddings_all = node_embedding[:self.user_num]
        item_embeddings_all = node_embedding[self.user_num:]

        # 如果聚类结果未初始化，则调用 e_step
        if self.user_2cluster is None or self.item_2cluster is None:
            self.e_step()
        
        # 如果调用后仍然为 None，则返回 0 损失（你也可以选择抛出异常或其他处理方式）
        if self.user_2cluster is None or self.item_2cluster is None:
            print("Warning: 聚类结果获取失败，ProtoNCE_loss 返回 0.")
            return 0.0

        # 用户部分
        user_embeddings = user_embeddings_all[user]  # [B, e]
        norm_user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
        # 索引聚类分配时确保 user 为 LongTensor
        user2cluster = self.user_2cluster[user]  # [B,]
        user2centroids = self.user_centroids[user2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # 物品部分
        item_embeddings = item_embeddings_all[item]  # [B, e]
        norm_item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
        item2cluster = self.item_2cluster[item]  # [B,]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss
