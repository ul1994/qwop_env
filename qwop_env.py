import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse

from tensorflow.python.client import timeline
import sys
import pytesseract
from qwop_config import config
# sys.path.append(config['pose_path'])
# from common import estimate_pose, CocoPairsRender, read_imgfile, CocoColors, draw_humans
# from networks import get_network
# from pose_dataset import CocoPoseLMDB
# import matplotlib.pyplot as plt

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# config.gpu_options.allow_growth = True

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pyautogui as gui
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

NUM_JOINTS = 7

GAME_SIZE = (1478, 924) # W, H
GAME_POS = (200, 250) # X, Y
POPUP_POS = (362, 290) # X, Y
DIST_POS = (440, 50) # X, Y
DIST_SIZE = (260, 80) # X, Y
# SCALE_PIX = 2

EXTRA_TERMS = 2

class QWOPEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 50
	}

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def get_screenshot(self):
		img = gui.screenshot(region=(GAME_POS[0],GAME_POS[1],GAME_SIZE[0],GAME_SIZE[1]))
		npimg = np.asarray(img, dtype=np.uint8)
		return npimg

	def crop_dist_img(self, img):
		return img[DIST_POS[1]:DIST_POS[1]+DIST_SIZE[1], DIST_POS[0]:DIST_POS[0]+DIST_SIZE[0], :]

	def __init__(self):
		# TODO: set discrete states
		self.action_space = spaces.Discrete(4)

		# first 14 represent x,y coords of body joints
		# 2nd to last is speed
		# last is total distance traveled
		self.total_states = NUM_JOINTS * 2 + EXTRA_TERMS

		high = np.zeros(self.total_states)
		high[:NUM_JOINTS] = 600.0 # First N are X coords
		high[NUM_JOINTS:NUM_JOINTS*2] = 400.0 # Last N are Y coords
		high[-2] = 10.0
		high[-1] = 105.0

		low = np.zeros(self.total_states)
		low[-2] = -2.0
		low[-1] = -2.0

		self.observation_space = spaces.Box(low, high)

		plt.ion()
		plt.figure(figsize=(6, 7))
		plt.show()

		pos_approved = False
		while not pos_approved:
			im = self.get_screenshot()
			plt.subplot(211)
			plt.imshow(im)
			plt.subplot(212)
			distIm = self.crop_dist_img(im)
			plt.imshow(distIm)
			plt.tight_layout()
			plt.show()
			plt.pause(0.01)
			pos_approved = input('[Is game in position? (y/n)]:').strip() == 'y'

		self._seed()
		self.reset()
		self.viewer = None
		self.state = None

	def reset_keypress(self):
		gui.press('r')

	def keypress(self, keynum, sleeptime = 0.2):
		keyVal = ['q', 'w', 'o', 'p']
		targKey = keyVal[keynum]
		gui.keyDown(targKey)
		time.sleep(sleeptime)
		gui.keyUp(targKey)

	def check_failPopup(self, img):
		check_pix = img[POPUP_POS[1], POPUP_POS[0], :]
		return check_pix[0] > 250 and check_pix[1] > 250

	def get_distance(self, img):
		distImg = self.crop_dist_img(img)
		dist = pytesseract.image_to_string(Image.fromarray(distImg), config='outputbase digits')
		dist = dist.split(' ')[0]
		return float(dist)

	def _step(self, action):
		currentImg = self.get_screenshot()
		distBefore = self.get_distance(currentImg)
		# TODO: get current distance
		self.keypress(action, sleeptime=0.2)
		# TODO: get distance travelled
		resultImg = self.get_screenshot()
		distAfter = self.get_distance(resultImg)

		diff = distAfter - distBefore
		reward = np.sign(diff) * (diff) ** 2.0

		didFall = self.check_failPopup(resultImg)
		done = didFall

		print ('%.1f > %.1f : Done-%s' % (distBefore, distAfter, str(done)))
		return np.array(self.state), reward, done, {}

	def _reset(self):
		# TODO: reset world states
		self.reset_keypress()
		self.state = np.zeros(self.total_states)
		return np.array(self.state)

	def _render(self, mode='human', close=False):
		"""
		We don't have to code a renderer b/c the game already exists.
		"""
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		screen_width = 600
		screen_height = 400

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)

		if self.state is None: return None

		return self.viewer.render(return_rgb_array = mode=='rgb_array')
