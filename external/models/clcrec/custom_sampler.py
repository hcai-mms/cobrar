import numpy as np
import random

class Sampler:
    def __init__(self, indexed_ratings, num_neg=128, seed=42):
        np.random.seed(seed)
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self.num_neg = num_neg
        self.all_set = set(self._items)
        self.user_item_dict = {u: set(indexed_ratings[u]) for u in indexed_ratings}

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            i = ui[r_int(lui)]

            j = random.sample(self.all_set - self.user_item_dict[u], self.num_neg) # list of negative items
            return u, i, j

        for batch_start in range(0, events, batch_size):
            bui, bii, bij = map(np.array, zip(*[sample() for _ in range(batch_start, min(batch_start + batch_size, events))]))
            yield bui[:, None], bii[:, None], bij[:, None]