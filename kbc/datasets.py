# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pkg_resources
import torch

from kbc.models import KBCModel

DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))


class Dataset(object):

    @staticmethod
    def get_dataset_shortlist():
        return ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', 'CKG-181019', 'CKG-181019-EXT']

    def __init__(self, name_or_path: str, use_cpu=False):
        shortlist = Dataset.get_dataset_shortlist()

        if name_or_path in shortlist:
            self.root = DATA_PATH / name_or_path
        else:
            self.root = name_or_path

        self.use_cpu = use_cpu

        self.entity_index = {}
        self.entity_reverse_index = {}
        self.relation_index = {}
        self.relation_reverse_index = {}

        # Loading entity mapping (format: <uri>\t[0-9]+)
        entity_map_file = open(os.path.join(self.root, "ent_id"), 'r')
        for line in entity_map_file.readlines():
            parts = line.split("\t")
            self.entity_index[parts[0]] = int(parts[1])
            self.entity_reverse_index[int(parts[1])] = parts[0]

        # Loading relation mapping (format: <uri>\t[0-9]+)
        rel_map_file = open(os.path.join(self.root, "rel_id"), 'r')
        for line in rel_map_file.readlines():
            parts = line.split("\t")
            self.relation_index[parts[0]] = int(parts[1])
            self.relation_reverse_index[int(parts[1])] = parts[0]

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(os.path.join(self.root, f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_node_id_from_name(self, name: str):
        return self.entity_index[name]

    def get_node_name_from_id(self, id: str):
        return self.entity_reverse_index[id]

    def get_rel_id_from_name(self, name: str):
        return self.relation_index[name]

    def get_rel_name_from_id(self, id: str):
        return self.relation_reverse_index[id]

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), batch_size=500
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64'))
        if not self.use_cpu:
            examples = examples.cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=batch_size)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor(list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            )))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
