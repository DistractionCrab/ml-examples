import torch
import tqdm

import ml.utils  as utils
import numpy     as np
import functools as ftools
import torch.nn  as nn


def generate_bits(m, n):
	a = np.random.choice((0.0, 1.0), size=(m,n))
	parities = list(int(sum(x) % 2) for x in a)
	b = np.eye(2)[np.array(parities)]
	return (torch.Tensor(a), torch.Tensor(b))

class ParityTrainer:
	def __init__(self, module):
		self.__model     = module
		self.__criterion = torch.nn.CrossEntropyLoss()
		self.__optimizer = torch.optim.Adam(
			self.model.parameters(), 
			lr=self.learning_rate)

	def train(self):
		for e in tqdm.tqdm(range(self.epochs), desc="Training..."):
			#print(f"--------- Epoch {e} -----------")
			(batch, labels) = generate_bits(self.batch_size, self.n_bits)
			predict = self.model(torch.Tensor(batch))

			loss = self.criterion(labels, predict)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		torch.save(self.model.state_dict(), self.path)


	def test(self):		
		(test_batch, test_labels) = generate_bits(10000, self.n_bits)
		predict = self.model(test_batch)

		# Check which labels are equal, which will have value 1 and sum up how many are 1.
		predict = torch.argmax(predict, dim=1)
		real = torch.argmax(test_labels, dim=1)
		sum = 0
		for (i, k) in zip(predict, real):
			if i == k:
				sum += 1
		#comp = np.multiply(test_labels.detach().numpy(), predict.detach().numpy()).sum()
		#numpy.einsum("ij", "ij->i", test_labels, predict)

		print(f"Accuracy of predictions: {sum/10000}")
		

	@property
	def batch_size(self):
		return 32
	
	@property
	def epochs(self):
		return 10000

	@property
	def learning_rate(self):
		return 0.001

	@property
	def regularization_beta(self):
		return 1e-5
	
	@property
	def model(self):
		return self.__model

	@property
	def criterion(self):
		return self.__criterion

	@property
	def optimizer(self):
		return self.__optimizer

	@property
	def n_bits(self):
		return self.__model.n_bits

	@property
	def path(self):
		return self.model.path
	
	


class ParityModule(torch.nn.Module):
	def __init__(self, n):
		super().__init__()
		self.__n_bits = n
		self.init()

	@property
	def n_bits(self):
		return self.__n_bits
	
		

	def init(self):
		b = self.n_bits

		# Chose network size that relates to b, this is not optimal, so should be played with.
		self.l1 = torch.nn.Linear(b, 3*b)
		self.r1 = torch.nn.Sigmoid()
		self.l2 = torch.nn.Linear(3*b, b)
		self.r2 = torch.nn.Sigmoid()
		self.l3 = torch.nn.Linear(b, b)
		self.r3 = torch.nn.Sigmoid()
		self.l4 = torch.nn.Linear(b, 2)
		self.r4 = torch.nn.Softmax(dim=1)

	def forward(self, x):
		out = self.l1(x)
		out = self.r1(out)
		out = self.l2(out)
		out = self.r2(out)
		out = self.l3(out)
		out = self.r3(out)
		out = self.l4(out)
		out = self.r4(out)

		return out

	@property
	def path(self):
		return utils.model_path(self.n_bits)

def train(n=3):
	model = ParityModule(n)
	ParityTrainer(model).train()

def test(n=3):
	model = ParityModule(n)	
	trainer = ParityTrainer(model)
	model.load_state_dict(torch.load(trainer.path))

	trainer.test()

utils.run([train, test])