import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class XSimGCL(LightGCN):
    """
    XSimGCL 模型在 LightGCN 的基础上，通过单次前向传播产生两个视图：
      - 最终层的嵌入：用于推荐任务（BPR 损失）
      - 某一中间层（layer_cl）的嵌入：用于对比学习损失
    """
    def __init__(self, data_handler):
        super(XSimGCL, self).__init__(data_handler)
        # 额外的超参数
        self.eps = self.hyper_config['eps']                   # 扰动幅度

        # 指定用于对比学习的中间层（例如第 layer_cl 层）的嵌入作为对比视图
        #self.layer_cl = self.hyper_config.get('layer_cl', self.layer_num - 1)
        self.layer_cl = self.hyper_config['layer_cl']
        self.cl_weight = self.hyper_config['cl_weight']       # 对比损失权重
        self.temperature = self.hyper_config['temperature']   # 温度系数

    def forward(self, perturbed=False):
        """
        前向传播：
        - 拼接用户和物品初始嵌入（存储在 self.embedding_dict 中）；
        - 使用 LightGCN 中的 _propagate 进行稀疏邻接矩阵传播；
        - 如果 perturbed=True，在每一层传播后添加随机扰动；
        - 在指定层保存输出作为对比学习视图；
        - 将所有层的嵌入取均值作为最终嵌入，再拆分成用户和物品嵌入。
        """
        # 初始嵌入（user_emb 与 item_emb 拼接）
        ego_embeddings = t.concat([self.user_embeds, self.item_embeds], dim=0)
        all_embeddings = []
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
        #all_embeddings_cl = t.stack([all_embeddings[1], all_embeddings[self.layer_cl]], dim=1).mean(dim=1)


        # 对所有层的嵌入取均值作为最终嵌入
        final_embeddings = t.stack(all_embeddings, dim=1).mean(dim=1)        # 拆分为用户和物品嵌入
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
         - 加上正则化项，形成总损失。
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

        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
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
