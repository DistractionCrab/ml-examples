import torch
import torch.autograd as ag
import functools      as ftools

class Cascade(torch.nn.Module):
	def __init__(self, in_size, neurons, out_size, act, train=False):
		super().__init__()
		self.__in_size = in_size
		self.__out_size = out_size
		self.__act = act
		self.__neurons = neurons

		# Weights used for inputs into each neuron.
		self.__nlinear = torch.nn.Parameter(
			torch.randn(in_size, neurons+out_size),
			requires_grad=train)

		self.register_parameter("cascade-linear", self.__nlinear)

		self.__nonlinear = [
			torch.nn.Parameter(
				torch.randn(i+1),
				requires_grad=train)
			for i in range(neurons-1)
		]

		for i, p in enumerate(self.__nonlinear):
			self.register_parameter(f"cascade-nonlinear{i}", p)

		self.__output_weights = [
			torch.nn.Parameter(torch.randn(neurons))
			for _ in range(neurons)
		]

		for i, p in enumerate(self.__output_weights):
			self.register_parameter(f"cascade-output-weights-{i}", p)



	def __forward_single(self, lin):

		# Recursive algorithm starting from the linear input to the first neuron. For each
		# neuron, including the output neuron, we apply an activation
		def recurse(acc, i):
			if i + 1 == self.__neurons: 
				return acc

			v = self.__act(acc)
			v = torch.sum(torch.matmul(v, self.__nonlinear[i])).expand(1)
			v = v + lin[i:i+1] + 1
			return recurse(torch.cat((acc, v)), i+1)

		

		t = recurse(lin[0:1] + 1, 0)
		


		# apply recursive to each of the last parts of lin, i.e. the linear inputs to each output
		# vector.
		outs = [
			torch.matmul(self.__act(t), self.__output_weights[i]) + l + 1
			for i, l in enumerate(lin[self.__neurons:]) 
		]

		return torch.stack(outs)

	# Input dimensions should be (batch, in_size)
	def forward(self, input):
		# Comput the linear inputs to each node from the primary inputs.
		lin = torch.matmul(input, self.__nlinear)

		values = [self.__forward_single(l) for l in lin]

		return torch.stack(values)

