import ml.networks.a2c as a2c
import vizdoom as vzd
import os

class Environment:
	def __init__(self):
		self.__game = vzd.DoomGame()