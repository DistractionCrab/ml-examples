import sys
import inspect
import ml
import pathlib
import shutil
import torch
import functools as ftools
import math

def run(fs):
	"""
	Function to be used by __main__ modules to call a set of functions specified by fs. Useful
	for calling modules that allow access to a set of functions to be called, and is less verbose.
	"""

	frm = inspect.stack()[1]
	mod = inspect.getmodule(frm[0])

	if mod.__name__ == "__main__":
		if len(sys.argv) > 1:
			fs = [f for f in fs if f.__name__ == sys.argv[1]]

			for f in fs:
				f(*sys.argv[2:])
		else:
			print("No function specified to run.")


MODEL_PATH = pathlib.Path(*pathlib.Path(ml.__file__).parts[:-2])/'models'

def model_path(append=None):
	frm = inspect.stack()[1]
	mod = inspect.getmodule(frm[0])

	if mod.__spec__ is None:
		raise NotImplementedError("Need to implement non-main module model path finding.")
		#name = mod.__name__.split('.')
		#fname = name[-1] + (f"_{append}.model" if len(append) > 0 else ".model")
	else:
		if '__main__' in mod.__spec__.name:
			name = mod.__spec__.name.split('.')[:-1]			
		else:			
			name = mod.__spec__.name.split('.')
			

		fname = name[-1] + (f"_{append}.model" if append is not None else ".model")

	path = ftools.reduce(lambda acc, x: acc/x, name[:-1], MODEL_PATH)

	if not path.exists():
		path.mkdir(parents=True)

	path /= fname	

	return path

def clear_models():
	shutil.rmtree(pathlib.Path(MODEL_PATH))


def bump_fn(var):
	# computes (x - 1)^2 * (x + 1)^2 * e^(-x**2) for -1 <= x <= 1, and 0 otherwise.
	return torch.where(
		var**2 <= 1, 
		((var - 1)**2)*((var + 1)**2)*torch.exp(-var**2),
		0)

class BumpNetwork(torch.nn.Module):
	def __init__(self, in_size, layers, out_size, b_count=10, train=False, act=lambda x: x):
		super().__init__()
		self.__in_size = in_size
		self.__layers = list(layers)
		self.__out_size = out_size
		self.__b_count = b_count
		self.__act = act

		self.__parameters = [
			torch.nn.Parameter(
				torch.FloatTensor(o, b_count, i).uniform_(-1.0, 1),
				requires_grad=train)
			for (i, o) in zip([in_size, *layers], [*layers, out_size])
		]

		for i, p in enumerate(self.__parameters):
			self.register_parameter(f"layers-{i}", p)

		self.__radii = [
			torch.nn.Parameter(torch.ones((b_count)))
			for _ in range(len(layers) + 1)
		]

		for i, p in enumerate(self.__radii):
			self.register_parameter(f"radii-{i}", p)

	def forward(self, inp):
		s = tuple(inp.size())

		if len(s) == 1:
			value = torch.nn.Tanh()(inp)
			for p, r in zip(self.__parameters, self.__radii):			
				value = p - value
				value = torch.linalg.norm(value, dim=2)/2
				value = torch.mul(value, r)
				value = bump_fn(value)
				value = value.mean(dim=1)

			return self.__act(value)
		else:
			values = [self.forward(i) for i in inp]

			return torch.stack(values)

		