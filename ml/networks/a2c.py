import torch
import torch.nn as nn
import gaime.networks.features as ft
import functools as ftools
import torch.nn.functional as fn
import math
import torch.optim as optim

from torch.distributions import Categorical
from dataclasses import dataclass
from collections import deque

@dataclass
class Params:
	features: ft.CVParams = ft.CVParams()
	# Can be various values from ft. e.g. ft.LSTMParams
	features_type: object = None
	layers: int = 1
	act: object = lambda i: torch.nn.ReLU()
	discount: float = 0.001


@dataclass
class TrainParams:
	epoch_length: int = 1000

class A2C_Vis(nn.Module):
	"""
	Sample A2C (Advantage Actor Critic) network that takes images for its input
	and generates output.
	"""

	def __init__(self, in_size, out_size, params=Params()):
		super().__init__()
		self.__params = params
		self.__in_size = in_size
		self.__out_size = out_size

		
		self.__features = ft.Image2d(
			in_size, 
			params=params.features,
			features=params.features_type)

		# Create a linearly decreasing size of linear layers.
		self.__actor = nn.Sequential()
		s = ftools.reduce(lambda x, a: x*a, self.__features.out_size, 1)
		sl = (out_size - s)/params.layers
		v = lambda x: math.ceil(sl * x + s)
		for i in range(params.layers):
			self.__actor.add_module(f"Actor-Linear {i}", nn.Linear(v(i), v(i+1)))
			self.__actor.add_module(f"Actor-Activation {i}", params.act(i))


		self.__critic = nn.Sequential()
		sl = (1 - s)/params.layers
		v = lambda x: math.ceil(sl * x + s)
		for i in range(params.layers):
			self.__critic.add_module(f"Critic-Linear {i}", nn.Linear(v(i), v(i+1)))
			self.__critic.add_module(f"Critic-Activation {i}", params.act(i))


	# Cache optimizers for specific networks. Generally no reason remake them each time.
	@ftools.lru_cache
	def optim(self, tparam=None):
		return optim.SGD(self.critic_parameters), optim.SGD(self.actor_parameters)

	def actor_parameters(self):
		yield from self.__features.parameters()
		yield from self.__actor.parameters()

	def critic_parameters(self):
		yield from self.__critic.parameters()

	def loss(self, actions, pis, crits, rws):
		discs = rws + self.__params.discount * crits[1:] - crits[:-1]
		exp = torch.log(pis)
		# Cap the log, so that products with 0 become 0.
		# This is about where torch.log's minimum gets to before becoming -inf.
		exp = torch.max(exp, torch.tensor(-100.0))
		exp = exp*discs

		return (discs**2).mean(), exp.sum()
		

	def forward(self, x):
		x = self.__features(x).flatten(start_dim=1)
		if self.training:
			return (self.__actor(x), self.__critic(x))
		else:
			return (self.__actor(x), None)

	def action(self, x):
		state, crit = self(x)

		distr = fn.softmax(state, dim=1) 
		choice = Categorical(distr).sample()
		return (choice, distr, crit)
		
	def train(self, env, tparam=None):
		if self.training:
			opt_c, opt_a = self.optim(tparam=tparam)

			actions, pis, crits, rws = [], [], [], []
			for step in range(tparam.epoch_length):
				(a, p, c) = self.action(env.state), 
				if env.finished:
					env.reset()
					crits.append(c)
					break
				else:				
					rwrd = env.step(a[0])
					actions.append(a)
					pis.append(p[0][a[0]])
					crits.append(c)
					rws.append(rwrd)

			actions = torch.cat(actions)
			pis = torch.stack(pis)
			crits = torch.stack(crits)
			rws = torch.stack(rws)
			lc, la = self.loss(actions, pis, crits, rws)

			opt_c.zero_grad()
			opt_a.zero_grad()
			lc.backward(retain_graph=True)
			la.backward(retain_graph=True)
			opt_c.step()
			opt_a.step()


