# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tqdm
import torch
from torch import nn
from torch import optim

from kbc.models import KBCModel
from kbc.regularizers import Regularizer


class KBCOptimizer(object):
    def __init__(self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
                 verbose: bool = True, use_cpu=False):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_cpu = use_cpu

    def epoch(self, examples: torch.LongTensor, epoch_number = -1, max_epochs = -1):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            description = f'train loss'
            description = "/ " + str(max_epochs) + " " + description
            description = str(epoch_number) + " " + description
            bar.set_description(description)
            b_begin = 0

            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ]
                if not self.use_cpu:
                    input_batch = input_batch.cuda()

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
