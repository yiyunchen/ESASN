import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import numpy as np

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(1.0 * params / (1000))

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

