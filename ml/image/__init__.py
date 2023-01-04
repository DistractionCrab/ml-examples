import numpy
import operator
import random
import torch
import torchvision
import tqdm

import ml.utils             as utils
import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

# How large each image in the dataset is.
MNIST_IMAGE_SIZE = (28, 28)
# The number of classes for thee MNIST dataset
MNIST_NUM_CLASSES = 10

class Trainer:
	def __init__(self, model):
		self.__model = model

		# Data for training the network
		self.__data_train = datasets.MNIST(
			'./.data', 
			train=True,
			download=True,
			transform=self.data_transform,
			target_transform=self.data_target_transform)

		# The data for testing how good our model is.
		self.__data_test = datasets.MNIST(
			'./.data', 
			train=False,
			download=True,
			transform=self.data_transform,
			target_transform=self.data_target_transform)

		self.__train_loader = torch.utils.data.DataLoader(
			dataset=self.__data_train, 
			batch_size=self.batch_size, 
			shuffle=True)
		self.__test_loader = torch.utils.data.DataLoader(
			dataset=self.__data_test, 
			batch_size=self.batch_size, 
			shuffle=False)

	def train(self):
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(
			self.model.parameters(), 
			lr=self.learning_rate)

		pbar = tqdm.trange(
			self.epochs, 
			desc="Average Loss: ????, Accuracy: ????%",
			position=0)
		
		for i in pbar:
			self.model.train()
			lsum = 0.0
			for (images, classes) in tqdm.tqdm(self.__train_loader, desc=f"Epoch: {i}", position=1):
				# Use the model to guess which class the image is.
				cl = self.model(images)

				loss = criterion(cl, classes)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				lsum += loss.item()

			# In real applications, you would do this more seldom.
			acc = self.test()
			pbar.set_description(
				f'Average Loss: {lsum/self.epochs:9.3f}, Accuracy: {acc*100:3.3f}%')

		torch.save(self.model.state_dict(), self.path)


	def test(self):
		self.model.eval()
		correct = 0
		inputs = torch.stack(list(map(lambda x: x[0], self.__data_test)))
		outputs = list(map(lambda x: x[1], self.__data_test))
		guesses = self.model(inputs)
		for (g, a) in zip(guesses, outputs):
			if g.argmax() == a:
				correct += 1
		return float(correct)/len(self.__data_test)
		

	@property
	def data_transform(self):
		return torchvision.transforms.Lambda(lambda i: torch.tensor(i.getdata(), dtype=torch.float))
		
	@property
	def data_target_transform(self):
		return lambda x: x
		#return lambda t: F.one_hot(torch.tensor(t), num_classes=NUM_CLASSES).float()

	@property
	def model(self):
		return self.__model
	
	@property
	def epochs(self):
		return 6


	@property
	def batch_size(self):
		return 128

	@property
	def learning_rate(self):
		return 0.001

	@property
	def regularization_beta(self):
		return 1e-5

	@property
	def path(self):
		raise NotImplementedError("path not implemented for general trainer.")
	
	
	
	
# Good questions to ask yourself, and test:
# 
# What happens when:
# - the batch size increases?
# - the batch size is 1?
# - We turn off weight_decay in our optimizer?
# - use a learning rate of 0.01 instead of 0.001?