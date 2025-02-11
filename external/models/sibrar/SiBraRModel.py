from abc import ABC

import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import collections


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class SiBraRModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 lr,
                 input_dim,
                 # ToDo mid_layers,
                 emb_dim,
                 w_decay,
                 cl_weight,
                 cl_temperature,
                 sp_i_train_ratings, # see deepmatrixfactorization, this is used as interaction modality for both user and item
                 item_modalities, # list of strings
                 use_user_profile,
                 input_dropout,
                 norm_input_feat,
                 norm_sbra_input,
                 # item multimodal features should also include interactions
                 item_multimodal_features, # actual tensors
                 random_seed,
                 name="ItemSiBraR",
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
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.lr = lr
        self.w_decay = w_decay
        self.cl_weight = cl_weight
        self.cl_temperature = cl_temperature
        self._sp_i_train_ratings = sp_i_train_ratings
        self.item_modalities = item_modalities
        self.name = name
        self.use_user_profile = use_user_profile
        self.input_dropout = input_dropout
        self.norm_input_feat = norm_input_feat
        self.norm_sbra_input = norm_sbra_input

        self.modality_projection_layers = {}
        self.item_multimodal_features = item_multimodal_features

        self.item_embedding_modules = {}

        # these are layers adapting each feature to the input of the single branch
        # this should NOT include the interactions, which are treated the same way as features,
        # but are always the last one
        for m_id, m in enumerate(self.item_modalities):
            layers = [(f'{m}_as_embedding', torch.nn.Embedding.from_pretrained(
                        torch.tensor(self.item_multimodal_features[m_id], dtype=torch.float32, device=self.device)
                    ))]

            if self.input_dropout > 0.:
                layers.append((f'{m}_dropout', nn.Dropout(p=self.input_dropout)))
            if self.norm_input_feat:
                layers.append((f'{m}_norm_layer', L2NormalizationLayer(dim=1)))
            layers.append((f'{m}_projector', torch.nn.Linear(self.item_multimodal_features[m_id].shape[1], self.input_dim).to(self.device)))
            layers = collections.OrderedDict(layers)

            self.item_embedding_modules[m_id] = torch.nn.Sequential(layers)

            # self.item_embedding_modules[m_id] = torch.nn.Sequential(
            #     collections.OrderedDict([
            #         (f'{m}_as_embedding', torch.nn.Embedding.from_pretrained(
            #         torch.tensor(self.item_multimodal_features[m_id], dtype=torch.float32, device=self.device)
            #     )),
            #         # (f'{m}_norm_layer', L2NormalizationLayer(dim=1)),
            #         (f'{m}_projector', torch.nn.Linear(self.item_multimodal_features[m_id].shape[1], self.input_dim).to(self.device))
            # ]))
            # the last modality is interactions

        layers = [(f'profile_as_embedding', torch.nn.Embedding.from_pretrained(
                    torch.tensor(self._sp_i_train_ratings.T.toarray(), dtype=torch.float32, device=self.device)
                ))]
        if self.norm_input_feat:
            layers.append((f'profile_norm_layer', L2NormalizationLayer(dim=1)))
        layers.append((f'profile_projector', torch.nn.Linear(self.num_users, self.input_dim).to(self.device)))
        layers = collections.OrderedDict(layers)
        self.item_embedding_modules[len(self.item_modalities)] = torch.nn.Sequential(layers)

        # self.item_embedding_modules[len(self.item_modalities)] = torch.nn.Sequential(
        #     collections.OrderedDict([
        #         (f'profile_as_embedding', torch.nn.Embedding.from_pretrained(
        #             torch.tensor(self._sp_i_train_ratings.T.toarray(), dtype=torch.float32, device=self.device)
        #         )),
        #         (f'{m}_norm_layer', L2NormalizationLayer(dim=1)),
        #         (f'profile_projector', torch.nn.Linear(self.num_users, self.input_dim).to(self.device))
        #         ])
        #     )

        # this is the actual single branch, shared by all item modalities
        # ToDo add option to add hidden layers ([ 512, 512, 512, 256, 256 ])
        self.single_branch = torch.nn.Linear(self.input_dim, self.emb_dim).to(self.device)

        # Right now only supports interactions, only, for users
        if self.use_user_profile:
            self.user_embedding_module = torch.nn.Sequential(
                collections.OrderedDict([
                ('profile_as_embedding', torch.nn.Embedding.from_pretrained(
                        torch.tensor(self._sp_i_train_ratings.toarray(), dtype=torch.float32, device=self.device)
                )),
                ('profile_projector', torch.nn.Linear(self.num_items, self.emb_dim).to(self.device))
            ]))
        else:
            self.user_embedding_module = torch.nn.Embedding(self.num_users, self.emb_dim).to(self.device)

        # AdamW is what was used for SiBraR
        # with weight decay instead of l2 reg
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.w_decay)


    def get_user_representations(self, users):
        feature = self.user_embedding_module(users)
        return feature


    def get_item_representations(self, items):
        features = torch.zeros((*items.squeeze().shape, len(self.item_modalities) + 1, self.emb_dim)).to(self.device)
        for m_id, m in enumerate(self.item_modalities):
            # ToDo check for activations (ReLu)
            feature = self.item_embedding_modules[m_id](items)
            if self.norm_sbra_input:
                feature = nn.functional.normalize(feature, p=2, dim=-1)
            # ToDo add batch normalization
            feature = self.single_branch(feature)
            features[..., m_id, :] = feature.squeeze()

        # Interactions (same ToDo as above)
        ## Activations
        ## batch normalization
        m_id = len(self.item_modalities)
        feature = self.item_embedding_modules[m_id](items)
        if self.norm_sbra_input:
            feature = nn.functional.normalize(feature, p=2, dim=-1)
        feature = self.single_branch(feature)
        features[..., m_id, :] = feature.squeeze()

        return features # Shape is [batch_size, num_modalities, embedding_dimension]

    def forward(self, inputs, **kwargs):
        # gu and gi should be the user and item batches
        users, items = inputs

        user_tensor = torch.tensor(users).to(self.device)
        items_tensor = torch.tensor(items).to(self.device)

        u_repr = self.get_user_representations(user_tensor)
        i_repr = self.get_item_representations(items_tensor)
        # i_repr = torch.mean(i_repr, dim=-2)
        return u_repr, i_repr

    def loss_contrastive(self, contrastive_modality_reps):
        # shape is [num_users, 1 + n_negs, 2, embedding_dim]
        contrastive_modality_reps = contrastive_modality_reps.reshape(-1, contrastive_modality_reps.size(-2), contrastive_modality_reps.size(-1))
        logits = contrastive_modality_reps[:, 0, :] @ contrastive_modality_reps[:, 1, :].transpose(-2, -1) / self.cl_temperature

        # Positive keys are the entries on the diagonal, therefore these are the correct labels:
        # [batch_size, 0, 1, ..., batch_size - 1]
        labels = torch.arange(logits.shape[-1], device=contrastive_modality_reps.device)#.repeat(1, *logits.shape[:-2], 1)

        # Logits change depending on which modality we are "retrieving"
        logits_c_to_p = logits.reshape(-1, logits.shape[-1])
        logits_p_to_c = logits.transpose(-2, -1).reshape(-1, logits.shape[-1])

        # labels = labels.reshape(-1)
        x_y_loss = F.cross_entropy(logits_c_to_p, labels)
        y_x_loss = F.cross_entropy(logits_p_to_c, labels)

        contrastive_loss = x_y_loss + y_x_loss
        return contrastive_loss

    def predict(self, gu, gi, **kwargs):
        return torch.sigmoid(torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1)))

    def train_step(self, batch):
        user, pos, neg = batch
        # user, pos, neg = user.to(self.device), pos.to(self.device), neg.to(self.device)

        user_repr, pos_item_repres = self.forward(inputs=(user, pos))
        _, neg_item_repres = self.forward(inputs=(user, neg))

        sampled_modalities_ids = np.random.choice(len(self.item_modalities) + 1, 2, replace=False)

        pos_item_repres = pos_item_repres[..., sampled_modalities_ids, :] # shape is [num_users, 2, embedding_dim]
        neg_item_repres = neg_item_repres[..., sampled_modalities_ids, :]

        pos_item_repr = torch.mean(pos_item_repres, dim=-2)
        neg_item_repr = torch.mean(neg_item_repres, dim=-2)

        xu_pos = torch.sum(user_repr * pos_item_repr, 1)
        xu_neg = torch.sum(user_repr * neg_item_repr, 1)

        loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))
        # self.item_embedding_modules
        # self.single_branch
        # self.user_embedding_module
        # reg_loss = self.l_w * (1 / 2) * (self.item_embedding_modules.weight[user[:, 0]].norm(2).pow(2) +
        #                                  self.Gi.weight[pos[:, 0]].norm(2).pow(2) +
        #                                  self.Gi.weight[neg[:, 0]].norm(2).pow(2)) / float(batch[0].shape[0])
        # loss += reg_loss
        contrastive_modality_reps = torch.cat((pos_item_repres[:, None, :, :], neg_item_repres), 1) # shape is [num_users, 1 + n_negs, 2, embedding_dim]
        contrastive_loss = self.loss_contrastive(contrastive_modality_reps)
        loss += self.cl_weight * contrastive_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()


    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
