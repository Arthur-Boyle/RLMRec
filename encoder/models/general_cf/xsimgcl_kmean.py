import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class XSimGCL_Kmean(LightGCN):
    """
    XSimGCL 模型在 LightGCN 的基础上，通过单次前向传播产生两个视图：
      - 最终层的嵌入：用于推荐任务（BPR 损失）
      - 某一中间层（layer_cl）的嵌入：用于对比学习损失
    同时集成了基于聚类的 ProtoNCE 对比损失，通过对用户和物品嵌入进行 K-means 聚类，将原始的嵌入映射到原型空间中计算相似性得分。
    """
    def __init__(self, data_handler):
        super(XSimGCL_Kmean, self).__init__(data_handler)
        # 原有超参数
        self.eps = self.hyper_config['eps']                   # 扰动幅度
        self.layer_cl = self.hyper_config.get('layer_cl', self.layer_num - 1)
        self.cl_weight = self.hyper_config['cl_weight']         # 对比损失权重
        self.temperature = self.hyper_config['temperature']     # 温度系数

        # 新增聚类相关的超参数
        self.latent_dim = self.hyper_config['embedding_size']       # 嵌入向量维度
        self.k = self.hyper_config['num_clusters']                         # 聚类的类别数（k-means 中的 k）
        self.ssl_temp = self.hyper_config['ssl_temp']           # ProtoNCE 温度参数
        self.proto_reg = float(self.hyper_config['proto_reg'])  # ProtoNCE 正则项权重

        # 以下假设 user_embedding 和 item_embedding 在 LightGCN 中已注册，若名称不同需要做相应修改
        # self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        # self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        # 另外，需要确保 self.device 已定义，否则可以用如下方式自动获取：
        self.device = "cuda"  # 假定 user_embeds 已经在指定设备上

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None
    
    def e_step(self):
        """
        E-step：利用当前的用户和物品嵌入，通过 K-means 得到对应的原型（centroid）和簇分配
        """
        # 这里假设 self.user_embedding 和 self.item_embedding 已初始化且权重与 self.user_embeds 保持一致
        user_embeddings = self.user_embeds.detach().cpu().numpy()
        item_embeddings = self.item_embeds.detach().cpu().numpy()

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
        centroids = t.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = t.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def forward(self, perturbed=False):
        """
        前向传播：
        - 拼接用户和物品初始嵌入（存储在 self.embedding_dict 中）；
        - 使用 LightGCN 中的 _propagate 进行稀疏邻接矩阵传播；
        - 如果 perturbed=True，在每一层传播后添加随机扰动；
        - 在指定层保存输出作为对比学习视图；
        - 将所有层的嵌入取均值作为最终嵌入，再拆分成用户和物品嵌入。
        """
        # 初始嵌入（user_embeds 与 item_embeds 拼接）
        ego_embeddings = t.concat([self.user_embeds, self.item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        all_embeddings_cl = ego_embeddings  # 默认对比视图为初始嵌入

        for layer in range(self.layer_num):
            # 使用 LightGCN 中的 _propagate 方法，调用 self.adj 进行传播
            ego_embeddings = self._propagate(self.adj, ego_embeddings)
            
            if perturbed:
                # 添加随机扰动：生成相同形状的随机噪声，归一化后乘以 sign(ego_embeddings) 和 eps
                noise = t.rand_like(ego_embeddings).cuda()
                ego_embeddings += t.sign(ego_embeddings) * F.normalize(noise, dim=-1) * self.eps

            all_embeddings.append(ego_embeddings)
            
            # 在指定层保存对比视图
            if layer == self.layer_cl - 1:
                all_embeddings_cl = ego_embeddings

        # 对所有层的嵌入取均值作为最终嵌入
        final_embeddings = t.stack(all_embeddings, dim=1).mean(dim=1)
        user_all_embeddings, item_all_embeddings = final_embeddings[:self.user_num], final_embeddings[self.user_num:]
        user_all_embeddings_cl, item_all_embeddings_cl = all_embeddings_cl[:self.user_num], all_embeddings_cl[self.user_num:]

        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        """
        损失计算：
         - 通过 perturbed 模式前向传播生成扰动视图，获取两个嵌入视图：
              * 最终层嵌入（用于推荐任务）
              * 对比层嵌入（中间层输出，用于对比学习）
         - 针对采样的 (用户, 正样本物品, 负样本物品) 三元组：
              * 用最终嵌入计算 BPR 损失；
              * 用跨层（对比层 vs 最终层）嵌入计算 InfoNCE 损失；
         - 加上 ProtoNCE 损失和正则化项，形成总损失。
        """
        outputs = self.forward(perturbed=True)
        user_embeds_final, item_embeds_final, user_embeds_cl, item_embeds_cl = outputs
        
        # 从三个视图中抽取 (用户, 正样本物品, 负样本物品) 的嵌入
        anc_embeds_final, pos_embeds_final, neg_embeds_final = self._pick_embeds(user_embeds_final, item_embeds_final, batch_data)
        anc_embeds_cl, pos_embeds_cl, neg_embeds_cl = self._pick_embeds(user_embeds_cl, item_embeds_cl, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds_final, pos_embeds_final, neg_embeds_final) / anc_embeds_final.shape[0]
        cl_loss = (
            cal_infonce_loss(anc_embeds_cl, anc_embeds_final, user_embeds_final, self.temperature) +
            cal_infonce_loss(pos_embeds_cl, pos_embeds_final, item_embeds_final, self.temperature)
        )
        cl_loss = cl_loss / anc_embeds_cl.shape[0] * self.cl_weight

        cecenter_embedding = t.concat([self.user_embeds, self.item_embeds], dim=0)
        ancs, poss, negs = batch_data
        # 聚类自监督的 ProtoNCE 损失
        proto_nce_loss = self.ProtoNCE_loss(cecenter_embedding, 
                                            ancs,  # 假定batch_data中包含用户下标
                                            poss)  # 以及物品下标，根据实际情况调整

        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss + cl_loss + proto_nce_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'proto_nce_loss': proto_nce_loss}
        return loss, losses

    def full_predict(self, batch_data):
        """
        预测阶段使用无扰动的最终嵌入，并屏蔽训练集中的物品，
        得到最终的用户对物品匹配分数矩阵。
        """
        user_embeds, item_embeds = self.forward(perturbed=False)
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
        pos_score_user = t.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = t.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = t.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = t.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        proto_nce_loss_user = -t.log(pos_score_user / ttl_score_user).sum()

        # 物品部分
        item_embeddings = item_embeddings_all[item]  # [B, e]
        norm_item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
        item2cluster = self.item_2cluster[item]  # [B,]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = t.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = t.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = t.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = t.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -t.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

