from abc import ABC

import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch_geometric.utils import scatter_


class CLCRecModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 reg_weight,
                 cl_weight,
                 modalities,
                 multimodal_features,
                 random_seed,
                 name="CLCRec",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.cl_weight = cl_weight
        self.modalities = modalities

        self.dim_feat = 0
        for m_id, m in enumerate(self.modalities):
            self.dim_feat += multimodal_features[m_id].size(1)

        self.MLP = torch.nn.Linear(self.embed_k, self.embed_k).to(self.device)

        self.encoder_layer1 = torch.nn.Linear(self.dim_feat, 256).to(self.device)
        self.encoder_layer2 = torch.nn.Linear(256, self.embed_k).to(self.device)
        
        self.att_weight_1 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.rand((self.embed_k, self.embed_k)))).to(self.device)
        self.att_weight_2 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.rand((self.embed_k, self.embed_k)))).to(self.device)
        self.bias = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.rand((self.embed_k, 1)))).to(self.device)
        self.att_sum_layer = torch.nn.Linear(self.embed_k, self.embed_k).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode_features(self, modalities):
        feature = torch.tensor([]).to(self.device)

        for m_id, m in enumerate(modalities):
            m_norm = F.normalize(m, dim=1)
            feature = torch.cat((feature, m_norm), dim=1)

        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return feature

    def forward(self, inputs, **kwargs):
        gu, gi = inputs

        pos_item_tensor = item_tensor[:, 0].unsqueeze(1)
        pos_item_tensor = pos_item_tensor.repeat(1, 1 + self.num_neg).view(-1, 1).squeeze()
        
        user_tensor = user_tensor.view(-1, 1).squeeze()
        item_tensor = item_tensor.view(-1, 1).squeeze()

        feature = self.encoder()
        all_item_feat = feature[item_tensor-self.num_user]

        user_embedding = self.id_embedding[user_tensor]
        pos_item_embedding = self.id_embedding[pos_item_tensor]
        all_item_embedding = self.id_embedding[item_tensor]
        
        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(pos_item_embedding, dim=1)

        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(all_item_embedding.size(0), (int(all_item_embedding.size(0)*self.num_sample), )).cuda()
        all_item_input[rand_index] = all_item_feat[rand_index].clone()

        self.contrastive_loss_1 = self.loss_contrastive(head_embed, head_feat, self.temp_value)
        self.contrastive_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temp_value)

        reg_loss = ((torch.sqrt((user_embedding**2).sum(1))).mean()+(torch.sqrt((all_item_embedding**2).sum(1))).mean())/2
        self.result = torch.cat((self.id_embedding[:self.num_user+self.num_warm_item], feature[self.num_warm_item:]), dim=0)

    def loss_contrastive(self, tensor_anchor, tensor_all, temp_value):      
        all_score = torch.exp(torch.sum(tensor_anchor*tensor_all, dim=1)/temp_value).view(-1, 1+self.num_neg)
        all_score = all_score.view(-1, 1+self.num_neg)
        pos_score = all_score[:, 0]
        all_score = torch.sum(all_score, dim=1)
        self.mat = (1-pos_score/all_score).mean()
        contrastive_loss = (-torch.log(pos_score / all_score)).mean()
        return contrastive_loss

    def predict(self, gu, gi, **kwargs):
        return torch.sigmoid(torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1)))

    def train_step(self, batch):
        user, pos, neg = batch

        contrastive_loss, reg_loss = self.forward(user, pos)
        reg_loss = self.reg_weight * reg_loss
        loss = contrastive_loss + reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
