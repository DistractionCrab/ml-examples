import gym
import sys
import operator
import random
import torch
import torchvision
import tqdm
import numpy                as np
import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

import ml.utils as utils
import ml.utils.games as games
import ml.rl as rl

class Trainer(rl.Trainer):
	def _train_epoch(self, crit, opt):
		self.model.env.reset(random.randint(0, sys.maxsize))
		rewards = [0]
		sums = []

		for _ in tqdm.trange(self.iterations, desc="Playing the Game", position=1):
			if self.model.env.done:
				self.model.env.reset(random.randint(0, sys.maxsize))				
				#self.model.env.seed()
				sums.append(sum(rewards))
				rewards = [0]

			qvalue = self.model(self.model.env.obsv	)			
			(r, done) = self.model.env.step(qvalue.argmax().item())
			rewards[0] += r

			qnext = self.model(self.model.env.obsv)

			loss = crit(r + qnext.max(), qvalue.max())
			opt.zero_grad()
			loss.backward()
			opt.step()
			
		return torch.FloatTensor(sums)

	@property
	def iterations(self):
		return 5000

	def criterion(self):
		return torch.nn.MSELoss()

	def optimizer(self):
		return torch.optim.Adam(
			self.model.parameters(), 
			lr=self.learning_rate)
	


class CartpoleModule(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.__env = games.CartpoleV1()

		# This network is simple and will run, but will not solve the Cartpole Problem.
		self.l1 = torch.nn.Linear(ftools.reduce(operator.mul, self.__env.obsv_shape), 64)
		self.r1 = torch.nn.Sigmoid()
		self.l2 = torch.nn.Linear(64, 30)
		self.r2 = torch.nn.Sigmoid()
		self.l3 = torch.nn.Linear(30, self.__env.num_act)
		self.r3 = torch.nn.Softmax(dim=0)

	def forward(self, x):
		out = self.l1(x)
		out = self.r1(out)
		out = self.l2(out)
		out = self.r2(out)
		out = self.l3(out)
		return self.r3(out)

	@property
	def env(self):
		return self.__env


def train_cartpole():
	module = CartpoleModule()
	trainer = Trainer(module)

	trainer.train()

utils.run([train_cartpole])