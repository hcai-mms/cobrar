import typing as t
import os
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class EmotionCategoricalAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.emotion_feature_folder_path = getattr(ns, "emotion_categorical_features", None)

        self.users = users
        self.items = items

        self.map_ = self.get_map_from_features(items, self.emotion_feature_folder_path)
        self.items = self.items & set(self.map_.keys())

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "EmotionCategoricalAttributes"
        ns.object = self
        ns.feature_map = self.map_
        ns.features = list({f for i in self.items for f in ns.feature_map[i]})
        ns.nfeatures = len(ns.features)
        ns.private_features = {p: f for p, f in enumerate(ns.features)}
        ns.public_features = {v: k for k, v in ns.private_features.items()}
        return ns

    def get_map_from_features(self, items, path, threshold=-0.061):
        map_ = {}
        for item in items:
            feat = np.load(self.emotion_feature_folder_path + '/' + str(item) + '.npy')
            cat_feat_list = [i + 1 for i, x in enumerate(feat) if x > threshold] # keep 0 for padding
            map_[item] = cat_feat_list
        return map_