# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import sys
from typing import Dict

import torch
from torch import optim
from torch.nn import DataParallel

from kbc.datasets import Dataset
from kbc.models import CP, ComplEx
from kbc.optimizers import KBCOptimizer
from kbc.regularizers import F2, N3

big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', 'CKG-181019', 'CKG-181019-EXT']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    "--save-model", nargs=1, default=[''], dest="save_model",
    help="Save final model to specified directory")

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

def list_int(values):
    return [int(value.strip()) for value in values.split(",")]

parser.add_argument(
    '--parallel', nargs='+', default=[], type=list_int,
    help="Activate parallel GPU processing with specified GPU device ids"
)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
args = parser.parse_args()

save_model_path = args.save_model
save_model = len(save_model_path) > 0 and len(save_model_path[0].strip()) > 0
if save_model:
    save_model_path = save_model_path[0]
    if os.path.exists(save_model_path):
        save_model_path = save_model_path + os.sep + args.dataset+".pickle"
    else:
        print("Directory specified for --save-model doesn't exist!")

dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

print(dataset.get_shape())
model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
}[args.model]()

if len(args.parallel)>0:
    model = DataParallel(model, device_ids=args.parallel[0]).cuda()
else:
    device = 'cuda'
    model.to(device)

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]




optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.SparseAdam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)





def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
for e in range(args.max_epochs):
    cur_loss = optimizer.epoch(examples, epoch_number=e, max_epochs=args.max_epochs)

    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)

if save_model:
    print("Saving trained model to "+ save_model_path+ " ...")
    torch.save(model.state_dict(), save_model_path)

results = dataset.eval(model, 'test', -1)
print("\n\nTEST : ", results)
