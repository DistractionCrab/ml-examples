import torch
import torch.nn as nn
import functools as ftools
import torch.nn.functional as fn
import math
import torch.optim as optim


class ScaledDP(nn.Module):
	def forward(self, q, k, v):
		m = q.matmul(k.t())
		m = m/torch.sqrt(torch.tensor(q.size()[1]))

		m = fn.softmax(m, dim=1)

		return m.matmul(v)

class MultiAttention(nn.Module):
	def __init__(self, dmodel, heads, att=ScaledDP):
		super().__init__()
		if dmodel % heads != 0:
			raise ValueError("heads must have a divisor of dmodel.")

		self.__heads = heads
		self.__maps = [
			tuple(nn.Linear(dmodel//heads, dmodel//heads) for _ in range(3))
			for _ in range(heads)]
		self.__att = att()

		self.__out = nn.Linear(dmodel, dmodel)

	def forward(self, q, k, v):
		qs = torch.split(q, self.__heads, dim=1)
		ks = torch.split(k, self.__heads, dim=1)
		vs = torch.split(v, self.__heads, dim=1)

		splits = [
			(qm(qs[i]),	km(ks[i]), vm(vs[i]),
			)
			for i, (qm, km, vm) in enumerate(self.__maps)
		]

		attented = [self.__att(q, k, v) for (q, k, v) in splits]

		cat = torch.cat(attented, dim=1)
		return self.__out(cat)








