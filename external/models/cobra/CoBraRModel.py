from abc import ABC

import torch
import numpy as np
import random
from torch import nn


class CoBraRModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 embedding_dim,
                 user_mlp,
                 item_mlp,
                 collaborative_branch,
                 reg,
                 similarity,
                 max_ratings,
                 sp_i_train_ratings,
                 learning_rate=0.01,
                 mu=1.e-6,
                 random_seed=42,
                 name="CoBraR",
                 **kwargs):
        super().__init__(**kwargs)
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
        self.embedding_dim = embedding_dim
        self.user_mlp = user_mlp
        self.item_mlp = item_mlp
        self.reg = reg
        self.similarity = similarity
        self.max_ratings = max_ratings
        self._sp_i_train_ratings = sp_i_train_ratings
        self.mu = mu
        # User and item profiles (rows and cols in the interaction matrix)
        # as pre-trained embeddings
        self.collaborative_branch = collaborative_branch
        # User and item profiles (rows and cols in the interaction matrix)
        # as pre-trained embeddings

        self.user_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(self._sp_i_train_ratings.toarray(), dtype=torch.float32, device=self.device))
        layers = []
        self.user_mlp = [self.num_items] + self.user_mlp + [self.collaborative_branch[0]]# + [self.embedding_dim]
        for i, (d1, d2) in enumerate(zip(self.user_mlp[:-1], self.user_mlp[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.to(self.device)
            layers.append(layer)
            activation = nn.ReLU().to(self.device)
            layers.append(activation)
        self.user_embedding_module = torch.nn.Sequential(*layers)
        for user_param in self.user_embedding_module.parameters():
            user_param.requires_grad = True

        self.item_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(self._sp_i_train_ratings.T.toarray(), dtype=torch.float32, device=self.device))
        layers = []
        self.item_mlp = [self.num_users] + self.item_mlp + [self.collaborative_branch[0]]
        for i, (d1, d2) in enumerate(zip(self.item_mlp[:-1], self.item_mlp[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.to(self.device)
            layers.append(layer)
            activation = nn.ReLU().to(self.device)
            layers.append(activation)
        self.item_embedding_module = torch.nn.Sequential(*layers)
        for item_param in self.item_embedding_module.parameters():
            item_param.requires_grad = True

        layers = []
        if len(self.collaborative_branch) == 1:
            self.collaborative_branch = [self.collaborative_branch[0]] + [self.embedding_dim]
        else:
            self.collaborative_branch = self.collaborative_branch + [self.embedding_dim]
        for i, (d1, d2) in enumerate(zip(self.collaborative_branch[:-1], self.collaborative_branch[1:])):
            layer = nn.Linear(in_features=d1, out_features=d2)
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.to(self.device)
            layers.append(layer)
            if layers != len(self.collaborative_branch) - 2:
                activation = nn.ReLU().to(self.device)
                layers.append(activation)

        self.collaborative_branch = torch.nn.Sequential(*layers)
        for cb_param in self.collaborative_branch.parameters():
            cb_param.requires_grad = True
        self.cosine_func = nn.CosineSimilarity(dim=-1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.reg)

        self.name = 'CollaborativeBranchRecommender'

    def get_user_representations(self, users):
        feature = self.user_embedding(users)
        feature = self.user_embedding_module(feature)
        feature = self.collaborative_branch(feature)
        return feature

    def get_item_representations(self, items):
        feature = self.item_embedding(items)
        feature = self.item_embedding_module(feature)
        feature = self.collaborative_branch(feature)
        return feature
    def forward(self, inputs, **kwargs):
        users, items = inputs

        user_tensor = torch.tensor(users).to(self.device)
        # user_tensor = users.clone().detach() or the
        # same
        # with .requires_grad_(True)

        items_tensor = torch.tensor(items).to(self.device)

        u_repr = self.get_user_representations(user_tensor)
        i_repr = self.get_item_representations(items_tensor)
        return u_repr, i_repr


    def predict(self, start_user, stop_user, **kwargs):
        """
        Predict scores for users in the range [start_user, stop_user) over all items.
        """
        user = torch.arange(start_user, stop_user).to(self.device)
        item = torch.arange(self.num_items).to(self.device)
        user_repr, item_repr = self.forward(inputs=(user, item))
        user_repr, item_repr = user_repr.to(self.device), item_repr.to(self.device)
        if self.similarity == 'cosine':
            preds = self.cosine_func(user_repr[:, None, :], item_repr)
            preds[preds < self.mu] = self.mu
        else:
            preds = torch.mm(user_repr, item_repr.T)
            preds = torch.sigmoid(preds)
        return preds

    def train_step(self, batch):
        user, item, label = batch

        label = label / self.max_ratings
        user_repr, item_repr = self.forward(inputs=(user, item))

        if self.similarity == 'cosine':
            preds = self.cosine_func(user_repr, item_repr)
            preds[preds < self.mu] = self.mu
            # print(preds.shape)
        else:
            preds = torch.mm(user_repr, item_repr.T)
            preds = torch.sigmoid(preds)

        loss = nn.BCELoss()(preds, torch.FloatTensor(label).to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)

