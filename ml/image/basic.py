import numpy
import operator
import random
import torch
import torchvision



import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

import ml.utils as utils
import ml.image as img_nn

class Module(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.l1 = torch.nn.Linear(ftools.reduce(operator.mul, img_nn.MNIST_IMAGE_SIZE), 128)
		self.r1 = torch.nn.Sigmoid()
		self.l2 = torch.nn.Linear(128, 64)
		self.r2 = torch.nn.Sigmoid()
		self.l3 = torch.nn.Linear(64, img_nn.MNIST_NUM_CLASSES)
		self.r3 = torch.nn.Softmax(dim=0)

	def forward(self, x):
		#print(x.size())
		out = self.l1(x)
		out = self.r1(out)
		out = self.l2(out)
		out = self.r2(out)
		out = self.l3(out)
		return self.r3(out)

class Trainer(img_nn.Trainer):
	@property
	def path(self):
		return utils.model_path()
	

def train():
	m = Module()
	trainer = Trainer(m)
	trainer.train()

def test():
	model = Module()	
	trainer = Trainer(model)
	model.load_state_dict(torch.load(trainer.path))

	acc = trainer.test()

	print(f"Test Accuracy: {acc*100}%")

utils.run([train, test])