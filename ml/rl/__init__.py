import ml
import gym
import cv2
import operator
import random
import torch
import torchvision
import tqdm
import numpy                as np
import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

from collections import deque

class Trainer:
	def __init__(self, model):
		self.__model = model

	def train(self):
		pbar = tqdm.trange(self.epochs, desc=f"Epoch Progress", position=0)
		(crit, opt) = (self.criterion(), self.optimizer())

		for e in pbar:
			self.model.train()
			rs = self._train_epoch(crit, opt)			
			pbar.set_description(f'Average Reward for Epoch: {rs.mean()}')


	def _train_epoch(self, criterion, optimizer):
		"""
		Runs an epoch of training. All environments will be reset after this returns.
		"""
		raise NotImplementedError('Training not implemented for particular model.')


	def test(self):
		done = False
		rwrd = 0.
		self.model.eval()
		while not done:
			action = self.model(self.model.env.obsv).argmax().item()
			(r, done) = self.model.env.step(action)
			rwrd += 1
		print(f'Total Evaluation Reward: {rwrd}')

	def criterion(self):
		raise NotImplementedError("Subclass must define their criterion for training.")

	def optimizer(self):
		raise NotImplementedError("Subclass must define their optimizer for training.")

	@property
	def model(self):
		raise NotImplementedError('Subclass must define their model to be used.')
	
	@property
	def epochs(self):
		return 20

	@property
	def learning_rate(self):
		return 0.001

	@property
	def regularization_beta(self):
		return 1e-5
	
	@property
	def model(self):
		return self.__model
	