import torch
import torch.nn as nn
import functools as ftools
import torch.nn.functional as fn
import math
import torch.optim as optim
import ml.networks.attention as att

class SinusoidalPositionalEncoder:
	"""
	Sinusoidal Positional Encoder as specified in Attention is all you Need by
	Vaswani et al.
	"""
	def __init__(self, length, dmodel, factor=10000):

		self.__parity = length % 2 == 0
		self.__length = length
		# Filters used to alternate sine and cosine in our words.
		self.__cos_filter = torch.tensor([1, 0]).float().repeat(math.ceil(length/2))
		self.__sin_filter = torch.tensor([0, 1]).float().repeat(length//2)
		# Factor used to divide out: pos/(k^(i/dmodel)), here self.__factor is the
		# denominator.
		self.__factor = 1.0/torch.tensor(factor).float()**(torch.arange(length)/dmodel)
		self.__factor = self.__factor.expand(1, -1)


	def forward(self, v):
		"""
		v -- A WxD matrix, where W is the number of words, and D is the
			length/dimension of each word representation.
		"""
		# Position values from 0 to W
		pos = torch.arange(v.size()[0]).float()
		# Creates an Wx1 matrix
		pos = pos.expand(1, -1).t()
		# Create a matrix P_{pos, i}, an WxD matrix
		pos = pos.matmul(self.__factor)

		cos = self.__cos_filter * pos.cos()
		sin = self.__sin_filter * pos.sin()

		return v + cos + sin




class EncoderCell(nn.Module):
	def __init__(self, dmodel, heads):
		super().__init__()
		self.__dmodel = dmodel

		# Linear Layers to convert single matrix of values into each Q, K, V
		self.__qlin = nn.Linear(dmodel, dmodel)
		self.__klin = nn.Linear(dmodel, dmodel)
		self.__vlin = nn.Linear(dmodel, dmodel)

		self.__att = att.MultiAttention(dmodel, heads)

		self.__norm1 = nn.LayerNorm(dmodel)
		self.__ff = PositionWiseFF(dmodel, dmodel)
		self.__norm2 = nn.LayerNorm(dmodel)

	def forward(self, v):
		q, k, v = self.__qlin(v), self.__klin(v), self.__vlin(v)

		out = self.__att(q, k, v)
		out = self.__norm1(v + out)
		out2 = self.__ff(out)
		return self.__norm2(out + out2)


def PositionWiseFF(insize, outsize):
	return nn.Sequential(
		nn.Linear(insize, outsize),
		nn.ReLU(),
		nn.Linear(outsize, outsize))


class DecoderCell(nn.Module):
	"""
	DecoderCell Implementation. Note that masking is not implemented as this is
	intended to be a demo of transformer models. Since masking is intended for batched
	inputs, it is not necessary for our needs.
	"""
	def __init__(self, dmodel, heads):
		super().__init__()
		self.__dmodel = dmodel

		# Linear Layers to convert single matrix of values into each Q, K, V
		self.__qlin = nn.Linear(dmodel, dmodel)
		self.__klin = nn.Linear(dmodel, dmodel)
		self.__vlin = nn.Linear(dmodel, dmodel)

		# Linear maps for encoder outputs.
		self.__qlin2 = nn.Linear(dmodel, dmodel)
		self.__klin2 = nn.Linear(dmodel, dmodel)

		self.__att = att.MultiAttention(dmodel, heads)

		self.__norm1 = nn.LayerNorm(dmodel)
		self.__ff = PositionWiseFF(dmodel, dmodel)
		self.__norm2 = nn.LayerNorm(dmodel)
		self.__norm3 = nn.LayerNorm(dmodel)

	def forward(self, v, venc):
		q, k, v = self.__qlin(v), self.__klin(v), self.__vlin(v)

		out  = self.__att(q, k, v)
		out2 = self.__norm1(v + out)
		q, k = self.__qlin2(venc), self.__klin2(venc)
		out3 = self.__att(q, k, out2)
		out4 = self.__norm2(out3 + out2)
		out5 = self.__ff(out4)
		return self.__norm3(out5 + out4)


class Transformer(nn.Module):
	def __init__(self, vocabsize, dmodel, heads, stack=6, outlin=None):
		super().__init__()
		self.__vocabsize = vocabsize
		self.__stack = stack
		self.__dmodel = dmodel
		self.__heads = heads
		self.__outlin = nn.Linear(dmodel, vocabsize) if outlin is None else outlin
		self.__probs = nn.Softmax()


		self.__encoders = [EncoderCell(dmodel, heads) for _ in range(stack)]
		self.__decoders = [DecoderCell(dmodel, heads) for _ in range(stack)]

	def forward(self, v, t):
		"""
		v -- The positionally-embedded sentence.
		t -- The partially formed target-sentence
		"""

		for i in range(self.__stack):
			v = self.__encoders[i](v)
			t = self.__decoders[i](v, t)

		t = self.__outlin(t)
		return self.__probs(t)





