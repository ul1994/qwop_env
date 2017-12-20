# POSE EST MODULES
import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import argparse

from tensorflow.python.client import timeline

from common import estimate_pose, CocoPairsRender, read_imgfile, CocoColors, draw_humans
from networks import get_network
from pose_dataset import CocoPoseLMDB

# POSE EST SETUP
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


# POSE EST MODEL
input_node = tf.placeholder(tf.float32, shape=(1, 368, 368, 3), name='image')
with tf.Session(config=config) as sess:
	net, _, last_layer = get_network('mobilenet', input_node, sess)

def pose_img(image):
	a = time.time()
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	pafMat, heatMat = sess.run(
		[
			net.get_output(name=last_layer.format(stage=6, aux=1)),
			net.get_output(name=last_layer.format(stage=6, aux=2))
		], feed_dict={'image:0': [image]}, options=run_options, run_metadata=run_metadata
	)
	tl = timeline.Timeline(run_metadata.step_stats)
	ctf = tl.generate_chrome_trace_format()
	heatMat, pafMat = heatMat[0], pafMat[0]

	avg = 0
	for _ in range(10):
		a = time.time()
		sess.run(
			[
				net.get_output(name=last_layer.format(stage=6, aux=1)),
				net.get_output(name=last_layer.format(stage=6, aux=2))
			], feed_dict={'image:0': [image]}
		)
		avg += time.time() - a
	a = time.time()
	humans = estimate_pose(heatMat, pafMat)
	process_img = CocoPoseLMDB.display_image(image, heatMat, pafMat, as_numpy=True)
	return process_img
