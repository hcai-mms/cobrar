from abc import ABC
import itertools as it

import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn

EPS = 1e-10

ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}

class BiVAECFMModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 beta_kl,
                 item_features,
                 random_seed,
                 encoder_structure=[20],
                 k=10,
                 act_fn="tanh",
                 likelihood="pois",
                 name="BiVAE",
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items

        self.learning_rate = learning_rate
        self.beta_kl = beta_kl

        self.item_encoder_structure = [num_users] + encoder_structure
        self.user_encoder_structure = [num_items] + encoder_structure

        self.mu_theta = torch.zeros((self.item_encoder_structure[0], k)).to(self.device)  # n_items*k
        self.mu_beta = torch.zeros((self.user_encoder_structure[0], k)).to(self.device)  # n_users*k

        self.theta = (torch.randn(self.item_encoder_structure[0], k) * 0.01).to(self.device)
        self.beta = (torch.randn(self.user_encoder_structure[0], k) * 0.01).to(self.device)
        torch.nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)
        if self.act_fn is None:
            raise ValueError("Supported act_fn: {}".format(ACT.keys()))

        self.item_features = torch.FloatTensor(item_features)
        self.item_prior_encoder = nn.Linear(item_features.shape[1], k).to(self.device)

        # User Encoder
        self.user_encoder = nn.Sequential()
        for i in range(len(self.user_encoder_structure) - 1):
            self.user_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(self.user_encoder_structure[i], self.user_encoder_structure[i + 1]),
            )
            self.user_encoder.add_module("act{}".format(i), self.act_fn)

        self.user_encoder.to(self.device)

        self.user_mu = nn.Linear(self.user_encoder_structure[-1], k).to(self.device)  # mu
        self.user_std = nn.Linear(self.user_encoder_structure[-1], k).to(self.device)

        # Item Encoder
        self.item_encoder = nn.Sequential()
        for i in range(len(self.item_encoder_structure) - 1):
            self.item_encoder.add_module(
                "fc{}".format(i),
                nn.Linear(self.item_encoder_structure[i], self.item_encoder_structure[i + 1]),
            )
            self.item_encoder.add_module("act{}".format(i), self.act_fn)

        self.item_encoder.to(self.device)
        
        self.item_mu = nn.Linear(self.item_encoder_structure[-1], k).to(self.device)  # mu
        self.item_std = nn.Linear(self.item_encoder_structure[-1], k).to(self.device)

        user_params = it.chain(
            self.user_encoder.parameters(),
            self.user_mu.parameters(),
            self.user_std.parameters(),
        )

        item_params = it.chain(
            self.item_encoder.parameters(),
            self.item_mu.parameters(),
            self.item_std.parameters(),
        )

        item_params = it.chain(item_params, self.item_prior_encoder.parameters())

        self.u_optimizer = torch.optim.Adam(params=user_params, lr=self.learning_rate)
        self.i_optimizer = torch.optim.Adam(params=item_params, lr=self.learning_rate)

    def encode_user_prior(self, x):
        h = self.user_prior_encoder(x)
        return h

    def encode_item_prior(self, x):
        h = self.item_prior_encoder(x)
        return h

    def encode_user(self, x):
        h = self.user_encoder(x)
        return self.user_mu(h), torch.sigmoid(self.user_std(h))

    def encode_item(self, x):
        h = self.item_encoder(x)
        return self.item_mu(h), torch.sigmoid(self.item_std(h))

    def decode_user(self, theta, beta):
        h = theta.mm(beta.t())
        return torch.sigmoid(h)

    def decode_item(self, theta, beta):
        h = beta.mm(theta.t())
        return torch.sigmoid(h)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x, user=True, beta=None, theta=None):
        x = torch.tensor(x, device=self.device)
        if user:
            mu, std = self.encode_user(x)
            theta = self.reparameterize(mu, std)
            return theta, self.decode_user(theta, beta), mu, std
        else:
            mu, std = self.encode_item(x)
            beta = self.reparameterize(mu, std)
            return beta, self.decode_item(theta, beta), mu, std

    def loss(self, x, x_, mu, mu_prior, std, kl_beta):
        x = torch.tensor(x, device=self.device)
        # Likelihood
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))

        ll = torch.sum(ll, dim=1)

        # KL term
        kld = -0.5 * (1 + 2.0 * torch.log(std) - (mu - mu_prior).pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(kl_beta * kld - ll)

    def predict(self, start_user, stop_user, **kwargs):
        theta_u = self.mu_theta[start_user:stop_user]
        return self.decode_user(theta_u, self.mu_beta)

    def train_step(self, tx, ids, user=True):
        batch = tx[ids, :]
        batch = batch.A

        if not user:
            beta, i_batch_, i_mu, i_std = self.forward(batch, user=user, theta=self.theta)

            i_batch_f = self.item_features[ids].to(self.device)
            i_mu_prior = self.encode_item_prior(i_batch_f)

            loss = self.loss(batch, i_batch_, i_mu, i_mu_prior, i_std, self.beta_kl)

            self.i_optimizer.zero_grad()
            loss.backward()
            self.i_optimizer.step()

            beta, _, i_mu, _ = self.forward(batch, user=user, theta=self.theta)

            self.beta.data[ids] = beta.data
            self.mu_beta.data[ids] = i_mu.data

        else:
            theta, u_batch_, u_mu, u_std = self.forward(batch, user=user, beta=self.beta)

            u_mu_prior = 0.0
            loss = self.loss(batch, u_batch_, u_mu, u_mu_prior, u_std, self.beta_kl)

            self.u_optimizer.zero_grad()
            loss.backward()
            self.u_optimizer.step()

            theta, _, u_mu, _ = self.forward(batch, user=user, beta=self.beta)

            self.theta.data[ids] = theta.data
            self.mu_theta.data[ids] = u_mu.data

        return loss

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
