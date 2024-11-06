from abc import ABC

import torch
import numpy as np
import random
import torch.nn.functional as F


class CLCRecModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 num_neg,
                 num_sample,
                 temperature,
                 lr_lambda,
                 reg_weight,
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
        self.num_neg = num_neg
        self.num_sample = num_sample
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.lr_lambda = lr_lambda
        self.reg_weight = reg_weight
        self.modalities = modalities

        self.dim_feat = 0
        self.F = torch.nn.ParameterList()
        for m_id, m in enumerate(self.modalities):
            self.F.append(torch.nn.Embedding.from_pretrained(torch.tensor(
                multimodal_features[m_id], device=self.device, dtype=torch.float32),
                freeze=False))
            self.dim_feat += multimodal_features[m_id].shape[1]

        self.id_embedding = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.rand((self.num_users + self.num_items, self.embed_k)))).to(self.device)

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
            feat = self.F[m_id].weight.to(self.device)
            m_norm = F.normalize(feat, dim=1)
            feature = torch.cat((feature, m_norm), dim=1)

        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return feature

    def forward(self, inputs, **kwargs):
        user_tensor, pos_tensor, neg_tensor = inputs

        user_tensor = torch.tensor(user_tensor).to(self.device)
        pos_tensor = torch.tensor(pos_tensor).to(self.device)
        neg_tensor = torch.tensor(neg_tensor).to(self.device)

        item_tensor = torch.cat((pos_tensor, neg_tensor[:, 0]), dim=1)
        user_tensor = user_tensor.repeat(1, 1+self.num_neg).view(-1, 1).squeeze()
        pos_tensor = pos_tensor.repeat(1, 1+self.num_neg).view(-1, 1).squeeze()
        
        user_tensor = user_tensor.view(-1, 1).squeeze()
        item_tensor = item_tensor.view(-1, 1).squeeze()

        feature = self.encode_features(self.modalities)
        all_item_feat = feature[item_tensor-self.num_users]

        user_embedding = self.id_embedding[user_tensor]
        pos_item_embedding = self.id_embedding[pos_tensor]
        all_item_embedding = self.id_embedding[item_tensor]
        
        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(pos_item_embedding, dim=1)

        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(all_item_embedding.size(0), (int(all_item_embedding.size(0)*self.num_sample), )).to(self.device)
        all_item_input[rand_index] = all_item_feat[rand_index].clone()

        self.contrastive_loss_1 = self.loss_contrastive(head_embed, head_feat, self.temperature)
        self.contrastive_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.temperature)

        cl_loss = self.contrastive_loss_1*self.lr_lambda+(self.contrastive_loss_2)*(1-self.lr_lambda)
        reg_loss = ((torch.sqrt((user_embedding**2).sum(1))).mean()+(torch.sqrt((all_item_embedding**2).sum(1))).mean())/2

        return cl_loss, reg_loss

    def loss_contrastive(self, tensor_anchor, tensor_all, temperature):      
        all_score = torch.exp(torch.sum(tensor_anchor*tensor_all, dim=1)/temperature).view(-1, 1+self.num_neg)
        all_score = all_score.view(-1, 1+self.num_neg)
        pos_score = all_score[:, 0]
        all_score = torch.sum(all_score, dim=1)
        self.mat = (1-pos_score/all_score).mean()
        contrastive_loss = (-torch.log(pos_score / all_score)).mean()
        return contrastive_loss

    def predict(self, start_user, stop_user, **kwargs):
        return torch.matmul(self.id_embedding[start_user:stop_user].to(self.device),
                            torch.transpose(self.id_embedding[self.num_users:], 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch

        contrastive_loss, reg_loss = self.forward((user, pos, neg))
        reg_loss = self.reg_weight * reg_loss
        loss = contrastive_loss + reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
