import torch
import tqdm

import ml.utils  as utils
import ml.parity.base as base
import ml.utils.cascade as cascade
import numpy     as np
import functools as ftools
import torch.nn  as nn

class Module(cascade.Cascade):
	def __init__(self, n):
		super().__init__(n, 10, 2, torch.nn.Sigmoid())
		self.__n_bits = n
		self.__out_act = torch.nn.Softmax(dim=1)

	@property
	def n_bits(self):
		return self.__n_bits

	def forward(self, input):
		return self.__out_act(super().forward(input))

	@property
	def path(self):
		return utils.model_path(self.n_bits)


def train(n=3):
	model = Module(n)
	base.ParityTrainer(model).train()

def test(n=3):
	model = base.ParityModule(n)	
	trainer = base.ParityTrainer(model)
	model.load_state_dict(torch.load(trainer.path))

	trainer.test()

utils.run([train, test])