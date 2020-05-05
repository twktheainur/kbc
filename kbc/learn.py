# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import glob
import os
import re
from typing import Dict

import numpy
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
    "--save-checkpoints", nargs=1, default=[''], dest="save_checkpoints",
    help="Save checkpoints for each epoch of the model to specified directory")

parser.add_argument(
    "--cpu", action='store_true', default=False,
    help="Use CPU for training"
)

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
    '--gpus', nargs='+', default=[], type=list_int,
    help="Comma separated list of gpu(s) to use, if several are provided computation will be distributed in parallel."
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

use_cpu = not torch.cuda.is_available() or args.cpu

save_model_path = args.save_model
save_model = len(save_model_path) > 0 and len(save_model_path[0].strip()) > 0
if save_model:
    save_model_path = save_model_path[0]
    if os.path.exists(save_model_path):
        save_model_path = save_model_path + os.sep + args.dataset + ".pickle"
        if os.path.exists(save_model_path):
            print(
                "Already trained model pickle found: " + save_model_path + ", exitting, please remove the file to retain.")
            exit(1)
    else:
        save_model = False
        print("Directory specified for --save-model doesn't exist!")

save_checkpoints_path = args.save_checkpoints
save_checkpoints = len(save_checkpoints_path) > 0 and len(save_checkpoints_path[0].strip()) > 0
if save_checkpoints:
    save_checkpoints_path = save_checkpoints_path[0]
    if os.path.exists(save_checkpoints_path):
        save_checkpoints_path = f"" + save_checkpoints_path + os.sep + args.dataset + "_{epochnum}.pickle"
    else:
        save_checkpoints = False
        print("Directory specified for --save-checkpoints doesn't exist!")

dataset = Dataset(args.dataset, use_cpu=use_cpu)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

print(dataset.get_shape())
model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
}[args.model]()

checkpoint_epoch = -1
checkpoint_pickles = glob.glob(save_checkpoints_path.format(epochnum="*"))
if len(checkpoint_pickles) > 0:
    epoch_num_pattern = re.compile(".*_([0-9]+)\.pickle")
    numbers = []
    for checkpoint in checkpoint_pickles:
        matches = re.match(epoch_num_pattern, checkpoint)
        numbers.append(int(matches.group(1)))
    checkpoint_epoch = numpy.array(numbers).max(initial=0) + 1
    model.load_state_dict(torch.load(save_checkpoints_path.format(epochnum=str(checkpoint_epoch - 1))))



if len(args.gpus) > 0:
    ids = args.gpus[0]
    if len(ids) > 1:
        model = DataParallel(model, device_ids=ids).cuda()
    else:
        device = torch.device('cuda:'+str(ids[0]))
        model.to(device)
else:
    if not use_cpu:
        device = 'cuda'
    else:
        device = "cpu"
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

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size,use_cpu=use_cpu)


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
if checkpoint_epoch > -1:
    training_range = range(checkpoint_epoch, args.max_epochs)
    print("Resuming training at epoch " + str(checkpoint_epoch))
else:
    training_range = range(args.max_epochs)

for e in training_range:
    cur_loss = optimizer.epoch(examples, epoch_number=e, max_epochs=args.max_epochs)

    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, batch_size=args.batch_size))
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)
    if save_checkpoints:
        path = save_checkpoints_path.format(epochnum=e)
        print("Saving epoch {epochnum} model to {path} ...".format(epochnum=e, path=path))
        torch.save(model.state_dict(), path)

if save_model:
    print("Saving trained model to " + save_model_path + " ...")
    torch.save(model.state_dict(), save_model_path)

results = dataset.eval(model, 'test', -1)
print("\n\nTEST : ", results)
