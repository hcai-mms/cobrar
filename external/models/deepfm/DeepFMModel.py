import numpy as np
import random

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_users, num_items, num_features, embedding_dim, padding_idx=None):
        super(EmbeddingLayer, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.feature_embedding = nn.Embedding(num_features, embedding_dim, padding_idx=padding_idx)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.feature_embedding.weight)

    def forward(self, user_ids, item_ids, feature_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        feature_emb = self.feature_embedding(feature_ids)

        return user_emb, item_emb, feature_emb

class DeepFMModel(nn.Module):
    def __init__(self, num_users, num_items, num_features, embed_dim, hidden_units, n_layers, dropout_rate, learning_rate, random_seed):
        super(DeepFMModel, self).__init__()
        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.field_dims = num_users + num_items + num_features

        # Embedding for FM and Deep components
        self.embedding = EmbeddingLayer(num_users, num_items, num_features + 1, embed_dim, padding_idx=0).to(self.device)

        # Linear term for FM
        self.linear = nn.Embedding(self.field_dims, 1).to(self.device)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)

        # Deep Component
        deep_input_dim = embed_dim * (self.num_features + 2)
        deep_modules = []
        for l in range(n_layers):
            deep_modules.append(nn.Linear(deep_input_dim, hidden_units))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout_rate))
            deep_input_dim = hidden_units
        deep_modules.append(nn.Linear(hidden_units, 1))
        self.deep = nn.Sequential(*deep_modules).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, user, item, features):
        user = torch.tensor(user).to(self.device)
        item = torch.tensor(item).to(self.device)
        features = torch.tensor(features).to(self.device)

        user_emb, item_emb, feat_emb = self.embedding(user, item, features)

        feats = torch.cat((user_emb.unsqueeze(1), item_emb.unsqueeze(1), feat_emb), dim=1)

        # FM Linear Term
        linear_inputs = torch.cat((user.unsqueeze(1), item.unsqueeze(1), features), dim=1)
        linear_part = torch.sum(self.linear(linear_inputs), dim=1)

        # FM Interaction Term
        sum_squared = torch.sum(feats, dim=1) ** 2
        squared_sum = torch.sum(feats ** 2, dim=1)
        interaction_part = 0.5 * torch.sum(sum_squared - squared_sum, dim=1, keepdim=True)
        fm_part = linear_part + interaction_part

        # Deep Component
        deep_part = self.deep(feats.flatten(start_dim=1))

        return torch.sigmoid(fm_part + deep_part).squeeze()

    def predict(self, start_user, stop_user, item_feat, batch_size=1024, **kwargs):
        """
        Predict scores for users in the range [start_user, stop_user) over all items in batches.
        """
        users = torch.arange(start_user, stop_user)
        items = torch.arange(self.num_items)
        item_feat = torch.tensor(item_feat)

        preds = []

        for i in range(0, self.num_items, batch_size):
            batch_items = items[i:i + batch_size]
            users_expanded = users.unsqueeze(1).repeat(1, len(batch_items)).flatten().to(self.device)
            items_expanded = batch_items.unsqueeze(0).repeat(len(users), 1).flatten().to(self.device)
            item_feat_expanded = item_feat[i:i + batch_size].repeat(len(users), 1).to(self.device)

            batch_preds = self.forward(users_expanded, items_expanded, item_feat_expanded)
            batch_preds = batch_preds.view(len(users), len(batch_items))
            preds.append(batch_preds)

        preds = torch.cat(preds, dim=1)
        return preds

    def train_step(self, batch):
        user, item, features, label = batch
        preds = self.forward(user, item, features)
        loss = nn.BCELoss()(preds, torch.FloatTensor(label).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)

