import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.general_cf.simgcl import SimGCL
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCLP(SimGCL):
    """
    SimGCLP 在 SimGCL 基础上增加了投影头，用于将扰动生成的嵌入映射到对比学习的空间，
    从而获得更好的表示分离效果。对比损失在投影空间上计算，而推荐任务仍然使用原始嵌入。
    """
    def __init__(self, data_handler):
        super(SimGCLP, self).__init__(data_handler)
        # 使用投影头将嵌入映射到对比空间，通常投影头可以是一个简单的 MLP
        self.projection_dim = self.hyper_config.get('projection_dim', self.embedding_size)
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_size, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.embedding_size)
        )
        # 对投影层做初始化（与其它权重初始化保持一致）
        for m in self.projection_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, adj=None, perturb=False, project=False):
        """
        当 project=True 时，对输出的用户和物品嵌入应用投影头，
        用于计算对比损失；否则返回原始嵌入，用于 BPR 损失。
        """
        user_embeds, item_embeds = super(SimGCLP, self).forward(adj, perturb)
        if project:
            user_embeds = self.projection_head(user_embeds)
            item_embeds = self.projection_head(item_embeds)
        return user_embeds, item_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        # 生成两组带扰动的视图，并在投影空间计算对比损失
        user_proj1, item_proj1 = self.forward(self.adj, perturb=True, project=True)
        user_proj2, item_proj2 = self.forward(self.adj, perturb=True, project=True)
        # 同时生成一组无扰动的嵌入，用于 BPR 损失计算
        user_embeds, item_embeds = self.forward(self.adj, perturb=False, project=False)
    
        anc_proj1, pos_proj1, neg_proj1 = self._pick_embeds(user_proj1, item_proj1, batch_data)
        anc_proj2, pos_proj2, neg_proj2 = self._pick_embeds(user_proj2, item_proj2, batch_data)
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)
    
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        # 在对比损失中使用投影后的表示计算 InfoNCE 损失
        cl_loss = cal_infonce_loss(anc_proj1, anc_proj2, user_proj2, self.temperature) \
                + cal_infonce_loss(pos_proj1, pos_proj2, item_proj2, self.temperature)
        cl_loss /= anc_proj1.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss *= self.cl_weight
        loss = bpr_loss + reg_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        # 预测仍然基于无扰动的原始嵌入
        user_embeds, item_embeds = self.forward(self.adj, perturb=False, project=False)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds