import sys

import torch
from kbc.datasets import Dataset

from kbc import avg_both
from kbc.models import CP

args = sys.argv[1:]

dataset = Dataset(args[0], use_cpu=True)
model = CP(dataset.get_shape(), 50)
model.load_state_dict(
    torch.load(args[1], map_location=torch.device('cpu')))

print(avg_both(*dataset.eval(model, "test", 50000, batch_size=100)))
