import numpy
import operator
import random
import torch
import torchvision

import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

import ml.utils       as utils
import ml.image       as img_nn
import ml.image.basic as base



class Trainer(base.Trainer):
	@property
	def data_transform(self):
		t1 = torchvision.transforms.Lambda(lambda i: torch.tensor(i.getdata(), dtype=torch.float))
		t2 = torchvision.transforms.Lambda(lambda i: i/255.0)
		return torchvision.transforms.Compose([t1, t2])
		#return torchvision.transforms.Compose([	
		#	torchvision.transforms.Normalize((0.1307,), (0.3081,))
		#])

	
	
def train():
	m = base.Module()
	trainer = Trainer(m)
	trainer.train()

def test():
	model = base.Module()	
	trainer = Trainer(model)
	model.load_state_dict(torch.load(trainer.path))

	acc = trainer.test()

	print(f"Test Accuracy: {acc*100}%")

utils.run([train, test])