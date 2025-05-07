import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCNWithAttention(BaseModel):
    def __init__(self, data_handler):
        super(LightGCNWithAttention, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = self.hyper_config.get('keep_rate', 1.0)
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']

        # 注意力机制
        self.attn_proj = nn.Linear(self.embedding_size, 1)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        # 使用注意力机制进行加权求和
        attn_weights = []
        for embed in embeds_list:
            attn = self.attn_proj(embed).squeeze(-1) # (num_users + num_items)
            attn_weights.append(attn)
        attn_weights = t.stack(attn_weights, dim=0) # (num_layers + 1, num_users + num_items)
        attn_weights = F.softmax(attn_weights, dim=0).unsqueeze(-1) # (num_layers + 1, num_users + num_items, 1)

        final_embeds = t.stack(embeds_list, dim=0) # (num_layers + 1, num_users + num_items, embedding_size)
        final_embeds = t.sum(final_embeds * attn_weights, dim=0)

        self.final_embeds = final_embeds
        return final_embeds[:self.user_num], final_embeds[self.user_num:]

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds