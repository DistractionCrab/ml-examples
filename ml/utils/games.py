import ml
import gym
import cv2
import operator
import random
import torch
import torchvision
import numpy                as np
import functools            as ftools
import torch.nn             as nn
import torch.nn.functional  as F
import torchvision.datasets as datasets

from collections import deque


class CartpoleV1:
	def __init__(self, render=False):
		self.__env = gym.make('CartPole-v1', new_step_api=True)
		self.__obsv = self.__env.reset()
		self.__done = False
		
	def reset(self, seed=0):
		self.__done = False
		self.__obsv = self.__env.reset(seed=seed)

	@property
	def env(self):
		return self

	@property
	def obsv(self):
		return torch.from_numpy(self.__obsv.astype('float32'))
	
	@property
	def num_act(self):
		return 2

	@property
	def obsv_shape(self):
		return (4,)
	
	@property
	def done(self):
		return self.__done
	
	def step(self, action):
		a = self.__env.step(action)
		#print(a)
		(self.__obsv, reward, self.__done, _, _) = a
		return (reward, self.__done)

class MsPacman:
	def __init__(self, render=False):
		self.__env            = gym.make('MsPacman-v4', new_step_api=True)
		self.__env.frame_skip = 4
		self.__render         = render
		self.reset()

		if render:
			pygame.init()
			self.__env.render()
		
	def reset(self, seed=0):
		self.__done   = False
		self.__obsv   = self.__process(self.__env.reset(seed=seed))
		self.__frames = deque([self.__obsv]*self.frame_save, maxlen=self.frame_save)
		self.__rwrd   = deque([0.0]*self.frame_save, maxlen=self.frame_save)

	@property
	def frame_save(self):
		return 4
	

	@property
	def env(self):
		return self

	@property
	def obsv(self):
		array = np.stack(self.__frames).astype('float32')
		tensor = torch.from_numpy(array)
		return torch.reshape(tensor, (1, 4, 84, 84))

	@property
	def rwrd(self):
		return sum(self.__rwrd)	
	
	@property
	def num_act(self):
		return 8

	@property
	def obsv_shape(self):
		return (84, 84, 4)
	
	@property
	def resize_shape(self):
		return (84, 84)
	

	@property
	def done(self):
		return self.__done
	
	def step(self, action):
		(obsv, reward, done, _, _) = self.__env.step(action)
		self.__obsv = self.__process(obsv)

		self.__frames.append(self.__obsv)
		self.__rwrd.append(reward)

		return (self.rwrd, done)

	def __process(self, obsv):
		return cv2.cvtColor(cv2.resize(obsv, self.resize_shape), cv2.COLOR_RGB2GRAY)