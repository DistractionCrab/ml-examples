"""
This module contains models pertaining to feature extraction in Reinforcement Learning.
General Convolutional methods are used on visual inputs, such as that for Image2d which can
leverage other recent techniques for feature extraction. For example, the LSTM methods
from Everett et al. in "Motion Planning Among Dynamic, Decision-Making Agents with Deep 
Reinforcement Learning" is implemented here as well.
"""

import enum
import functools as ftools
import torch
import torch.nn as nn
import dataclasses

DEFAULT_HIDDEN_SIZE = 288


def lstm_size(i_size, params):
	w = i_size[1] - (params.kernel_size[0] - 1)*params.layers
	h = i_size[2] - (params.kernel_size[1] - 1)*params.layers
	return (params.filters, w//params.kernel_size[0], h//params.kernel_size[1])

@dataclasses.dataclass(frozen=True)
class LSTMParams:
	h_size: int

@dataclasses.dataclass
class CVParams:
	layers: int = 2
	filters: int = 32
	kernel_size: int = (2, 2)
	stride: int = 1
	padding: int = 0
	act: object = lambda i: nn.LeakyReLU()

	def __post_init__(self):
		if type(self.kernel_size) is int:
			self.kernel_size = (self.kernel_size, self.kernel_size)

class LSTM(nn.Module):
	def __init__(self, i_size, params):
		super().__init__()

		self.__h = torch.zeros(params.h_size)
		self.__c = torch.zeros(params.h_size)

		self.__lstm = nn.LSTMCell(input_size=i_size, hidden_size=params.h_size)

	def forward(self, x):
		h = self.__h
		c = self.__c
		for v in x:
			(h, c) = self.__lstm(v, (h, c))

		return h

class Image2d(nn.Module):
	"""
	Feature extraction network for 2D images. Can be a simple Convolution extraction
	or can leverage LSTM techniques. If no features are specified, then it will
	just be convolutional extraction. If LSTM is specified, batches of inputs will
	be treated in sequence to an LSTM cell as specified in Everett et al. to compute a
	feature vector. These batches can be a sequence of agent observables or can be
	sequential image information as is standard in RL.
	"""

	def __init__(self, i_size, params=CVParams(), features=None):
		super().__init__()
		self.__params = dataclasses.replace(params)
		# make sure i_size contains channel information. If only a 2-d image is given,
		# then make sure channels are saved as 1.
		self.__i_size = i_size if len(i_size) == 3 else (1, *i_size)
		self.__conv = nn.Sequential()

		self.__conv.add_module(			
			"Conv2d-layer: 0",
			nn.Conv2d(
				self.__i_size[0], 
				self.__params.filters,
				self.__params.kernel_size,
				stride=self.__params.stride,
				padding=self.__params.padding
		))
		a = self.__params.act(0)
		self.__conv.add_module(f"{a.__class__.__name__}: 0", a)

		for i in range(1, params.layers):
			self.__conv.add_module(
				f"Conv2d-layer: {i}",
				nn.Conv2d(
					self.__params.filters, 
					self.__params.filters,
					self.__params.kernel_size,
					stride=self.__params.stride,
					padding=self.__params.padding
			))
			a = self.__params.act(i)
			self.__conv.add_module(f"{a.__class__.__name__}: {i}", a)

		self.__conv.add_module("Max-Pooling", torch.nn.MaxPool2d(
			kernel_size=params.kernel_size))

		self.__fparam = features
		if type(features) is LSTMParams:
			s = lstm_size(self.__i_size, self.__params)
			s = ftools.reduce(lambda x, a: x*a, s, 1)
			self.__features = LSTM(s, features)
		else:
			self.__features = nn.Sequential()

	@property
	def out_size(self):		
		if type(self.__fparam) is LSTMParams:
			return torch.Size((self.__fparam.h_size,))
		else:
			return torch.Size(lstm_size(self.__i_size, self.__params))
	

	def forward(self, x):
		# Takes in image sequence of size (s, ch, w, h)
		# Output size will be (s, f, w-k[w], h-k[h])
		v = self.__conv(x)
		#print(v.size())
		v = torch.flatten(v, start_dim=1)
		v = self.__features(v)

		v = v.reshape(-1, *self.out_size)

		return v


		


