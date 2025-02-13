from abc import ABC

import torch
import numpy as np
import random
from torch import nn


class DeepMFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 user_mlp,
                 item_mlp,
                 reg,
                 similarity,
                 max_ratings,
                 sp_i_train_ratings,
                 learning_rate=0.01,
                 random_seed=42,
                 name="DeepMF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
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
        self.user_mlp = user_mlp
        self.item_mlp = item_mlp
        self.reg = reg
        self.similarity = similarity
        self.max_ratings = max_ratings
        self._sp_i_train_ratings = sp_i_train_ratings

        self.user_embedding = torch.nn.Embedding.from_pretrained(
                        torch.tensor(self._sp_i_train_ratings.toarray(), dtype=torch.float32, device=self.device))
        self.item_embedding = torch.nn.Embedding.from_pretrained(
                    torch.tensor(self._sp_i_train_ratings.T.toarray(), dtype=torch.float32, device=self.device))
        layers = []
        for i, (d1, d2) in enumerate(zip(user_mlp[:-1], user_mlp[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.to(self.device)
            layers.append(layer)
            if layers != len(user_mlp) - 2:
                activation = nn.ReLU().to(self.device)
                layers.append(activation)

        self.user_mlp_layers = torch.nn.Sequential(layers)

        for i, (d1, d2) in enumerate(zip(item_mlp[:-1], item_mlp[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.to(self.device)
            layers.append(layer)
            if layers != len(user_mlp) - 2:
                activation = nn.ReLU().to(self.device)
                layers.append(activation)

        self.item_mlp_layers = torch.nn.Sequential(layers)

        self.optimizer = torch.optim.Adam(lr=learning_rate)

    def get_user_representations(self, users):
        feature = self.user_embedding_module(users)
        return feature

    def get_item_representations(self, items):
        feature = self.item_embedding_module(items)
        return feature

    def forward(self, inputs, **kwargs):
        users, items = inputs

        user_tensor = torch.tensor(users).to(self.device)
        items_tensor = torch.tensor(items).to(self.device)

        u_repr = self.get_user_representations(user_tensor)
        i_repr = self.get_item_representations(items_tensor)
        return u_repr, i_repr


    def predict(self, gu, gi, **kwargs):
        return torch.sigmoid(torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1)))

    def train_step(self, batch):
        user, item, label = batch
        user_repr, item_repr = self.forward(inputs=(user, item))
        preds = torch.sum(user_repr * item_repr, 1)

        loss = nn.BCELoss()(preds, torch.FloatTensor(label).to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)