def forward(self):
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None, None, None, None

        all_embeds = [torch.concat([self.user_embeds.weight, self.item_embeds.weight], dim=0)]
        gnn_embeds, gaa_embeds = [], []

        for i in range(0, self.layer_num):
            # 使用 LightGCN 的传播方式进行图卷积
            gnn_layer_embeds = self._propagate(self.adj, all_embeds[i])


            # 增强视角
            gnn_head_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_h_list)
            gnn_tail_embeds = torch.index_select(gnn_layer_embeds, 0, self.all_t_list)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeds, gnn_tail_embeds)
            gaa_layer_embeds = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Aggregation
            gnn_embeds.append(gnn_layer_embeds)
            gaa_embeds.append(gaa_layer_embeds)
            all_embeds.append(gnn_layer_embeds + gaa_layer_embeds + all_embeds[i])

        all_embeds = torch.stack(all_embeds, dim=1)
        all_embeds = torch.sum(all_embeds, dim=1, keepdim=False)
        user_embeds, item_embeds = torch.split(all_embeds, [self.user_num, self.item_num], 0)
        self.final_embeds = all_embeds
        return user_embeds,item_embeds, gnn_embeds, gaa_embeds

def e_step(self):
        """
        E-step：利用当前的用户和物品嵌入，通过 K-means 得到对应的原型（centroid）和簇分配
        """
        # 这里假设 self.user_embedding 和 self.item_embedding 已初始化且权重与 self.user_embeds 保持一致
        user_embeddings = self.user_embeds.weight.detach().cpu().numpy()
        item_embeddings = self.item_embeds.weight.detach().cpu().numpy()


        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

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

def _cal_cl_loss(self, users, items, gnn_emb, gaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)  # different from original SSLRec, remove negative items
        cl_loss = 0.0
        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.user_num, self.item_num], 0)
            
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.user_num, self.item_num], 0)
            

            u_gnn_embs = u_gnn_embs[users]
            u_gaa_embs = u_gaa_embs[users]
            

            i_gnn_embs = i_gnn_embs[items]
            i_gaa_embs = i_gaa_embs[items]

            
            cl_loss += cal_infonce_loss(u_gnn_embs, u_gaa_embs, u_gaa_embs, self.temperature) / u_gnn_embs.shape[0]
            
            cl_loss += cal_infonce_loss(i_gnn_embs, i_gaa_embs, i_gaa_embs, self.temperature) / u_gnn_embs.shape[0]
        return cl_loss

def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds,item_embeds, gnn_embeds, gaa_embeds = self.forward()
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, gnn_embeds, gaa_embeds)
        cecenter_embedding = torch.concat([user_embeds, item_embeds], dim=0)
        ancs, poss, negs = batch_data
        # 聚类自监督的 ProtoNCE 损失
        proto_nce_loss = self.ProtoNCE_loss(cecenter_embedding, 
                                            ancs,  # 假定batch_data中包含用户下标
                                            poss)  # 以及物品下标，根据实际情况调整
        loss = bpr_loss + reg_loss + proto_nce_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
        return loss, losses

def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss