import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import random
from config import Config
from dataLoader import DataLoader

from nets import *
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='10'


class SfMLearner(object):
	def __init__(self, opt):
		self.opt = opt
	
	def preprocess_image(self, image):
		# Assume input image is uint8
		image = tf.image.convert_image_dtype(image, dtype = tf.float32)
		return image * 2. - 1.
	
	def deprocess_image(self, image):
		image = (image + 1.) / 2.
		return tf.image.convert_image_dtype(image, dtype = tf.uint8)


	def build_train_graph(self):
		dataloader = DataLoader(self.opt.dataset_dir,
								self.opt.batch_size,
								self.opt.img_height,
								self.opt.img_width,
								self.opt.seq_length,
								self.opt.num_source,
								self.opt.num_scales)

		with tf.name_scope("data_loading"):
			self.intrinsics, self.tgt_image, self.src_image_stack = \
				dataloader.load_train_batch()

			self.tgt_image = self.preprocess_image(self.tgt_image)
			self.src_image_stack = self.preprocess_image(self.src_image_stack)

		# pred_disp : [disp1, disp2, disp3, disp4]
		# disp1 : [B, H, W, 1]
		# disp2 : [B, H/2, W/2, 1]
		# disp3 : [B, H/4, W/4, 1]
		# disp4 : [B, H/8, W/8, 1]
		with tf.name_scope("depth_prediction"):
			self.pred_disp, self.depth_net_endpoints = disp_net(self.tgt_image,
														is_training=True)

			self.pred_depth = [(1. / disp ) for disp in self.pred_disp]
			"""
			for k, v in self.depth_net_endpoints.items():
				print(k, v)
			"""

		# self.pred_pose_avg : [B, 12]
		# self.pred_pose_final : [B, 2, 6]
		with tf.name_scope("pose_prediction"):
			self.pred_pose_avg, self.pred_pose_final, self.pred_mask, self.pose_net_endpoints = \
				pose_exp_net(self.tgt_image, self.src_image_stack, False, True)

			"""
			for k, v in self.pose_net_endpoints.items():
				print(k, v)
			print(self.pred_pose_avg)
			print(self.pred_pose_final)
			"""

		with tf.name_scope("compute_loss"):
			pixel_loss = 0
			smooth_loss = 0
			tgt_image_all = []
			src_image_stack_all = []
			proj_image_stack_all = []
			proj_error_stack_all = []

			for s in range(self.opt.num_scales):
				curr_tgt_image = tf.image.resize_area(self.tgt_image, 
													[int(self.opt.img_height / (2**s)), int (self.opt.img_width / (2**s))])
				curr_src_image_stack = tf.image.resize_area(self.src_image_stack, 
													[int(self.opt.img_height / (2**s)), int (self.opt.img_width / (2**s))])

				if self.opt.smooth_weight > 0:
					smooth_loss += self.opt.smooth_weight/(2**s) * \
						self.compute_smooth_loss(self.pred_disp[s])
				for i in range(self.opt.num_source):
					curr_proj_image = \
							proj_inverse_map(tf.slice(curr_src_image_stack, [0, 0, 0, i * 3], [-1, -1, -1, 3]), 
									 tf.squeeze(self.pred_depth[s], axis = 3),
									 tf.squeeze(tf.slice(self.pred_pose_final, [0, i, 0], [-1, 1, -1]), axis = 1),
									 #self.pred_pose_final[:, 1, :],
									 tf.squeeze(tf.slice(self.intrinsics, [0, s, 0, 0], [-1, 1, -1, -1]), axis = 1))
					curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
					
					pixel_loss += tf.reduce_mean(curr_proj_error)

					# Prepare images for tensorboard summaries
					if i == 0:
						proj_image_stack = curr_proj_image
						proj_error_stack = curr_proj_error
					else:
						proj_image_stack = tf.concat([proj_image_stack, curr_proj_image], axis = 3)

						proj_error_stack = tf.concat([proj_error_stack, curr_proj_error], axis = 3)
				tgt_image_all.append(curr_tgt_image)
				src_image_stack_all.append(curr_src_image_stack)
				proj_image_stack_all.append(proj_image_stack)
				proj_error_stack_all.append(proj_error_stack)
			total_loss = pixel_loss + smooth_loss

		with tf.name_scope("train_op"):
			train_vars = [var for var in tf.trainable_variables()]
			optim = tf.train.AdamOptimizer(self.opt.learning_rate, self.opt.beta1)
			self.train_op = slim.learning.create_train_op(total_loss, optim)
			self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
			self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

		self.steps_per_epoch = dataloader.steps_per_epoch
		self.total_loss = total_loss
		self.pixel_loss = pixel_loss
		self.smooth_loss = smooth_loss
		self.tgt_image_all = tgt_image_all
		self.src_image_stack_all = src_image_stack_all
		self.proj_image_stack_all = proj_image_stack_all
		self.proj_error_stack_all = proj_error_stack_all


			

	def compute_smooth_loss(self, pred_disp):
		def gradient(pred):
			D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
			D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
			return D_dx, D_dy
		dx, dy = gradient(pred_disp)
		dx2, dxdy = gradient(dx)
		dydx, dy2 = gradient(dy)
		return tf.reduce_mean(tf.abs(dx2)) + \
				tf.reduce_mean(tf.abs(dxdy)) + \
				tf.reduce_mean(tf.abs(dydx)) + \
				tf.reduce_mean(tf.abs(dy2))

	def collect_summaries(self):
		tf.summary.scalar("total_loss", self.total_loss)
		tf.summary.scalar("pixel_loss", self.pixel_loss)
		tf.summary.scalar("smooth_loss", self.smooth_loss)
		for s in range(self.opt.num_scales):
			tf.summary.histogram("scale%d_depth" %s, self.pred_depth[s])
			tf.summary.image("scale%d_disparity_image" %s, 1./self.pred_depth[s])
			tf.summary.image("scale%d_tgt_image" % s, \
							self.deprocess_image(self.tgt_image_all[s]))
			for i in range(self.opt.num_source):
				tf.summary.image('scale%d_source_image_%d' %(s, i),
								 self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
				tf.summary.image('scale%d_projected_image_%d' %(s, i),
								 self.deprocess_image(self.proj_image_stack_all[0][:, :, :, i *3:(i+1)*3]))
				tf.summary.image('scale%d_proj_error_%d' %(s, i),
								 self.deprocess_image(tf.clip_by_value(self.proj_error_stack_all[s][:, :, :, i * 3: (i + 1) *3] - 1, -1, 1)))
		tf.summary.histogram('tx', self.pred_pose_final[:, :, 0])
		tf.summary.histogram('ty', self.pred_pose_final[:, :, 1])
		tf.summary.histogram('tz', self.pred_pose_final[:, :, 2])
		tf.summary.histogram('rx', self.pred_pose_final[:, :, 3])
		tf.summary.histogram('ry', self.pred_pose_final[:, :, 4])
		tf.summary.histogram('rz', self.pred_pose_final[:, :, 5])

		
	
	def train(self):
		self.build_train_graph()
		self.collect_summaries()

		"""
		for var in tf.model_variables():
			print(var)
		"""

		self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
									[self.global_step],
									max_to_keep = 10)
		
		sv = tf.train.Supervisor(logdir=self.opt.checkpoint_dir,
								 save_summaries_secs = 0,
								 saver = None)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		with sv.managed_session(config = config) as sess:
			#print(sess.run(self.total_loss))
			start_time = time.time()
			fetches = {	"loss" : self.total_loss,
						"train" : self.train_op,
						"global_step" : self.global_step,
						"incr_global_step" : self.incr_global_step}
			for step in range(1, self.opt.max_steps):
				if step % self.opt.summary_freq == 0 :
					fetches["summary"] = sv.summary_op

				results = sess.run(fetches)
				gs = results["global_step"]

				if step % self.opt.summary_freq == 0 :
					sv.summary_writer.add_summary(results["summary"], gs)
					print("step: [%2d], loss:[%3f], time: %4f" %(step, results['loss'], (time.time() - start_time)))
					start_time = time.time()

				if step % self.steps_per_epoch == 0:
					self.save(sess, self.opt.checkpoint_dir, gs)
	
	def save(self, sess, checkpoint_dir, step):
		model_name = 'model'
		self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step = step)
	


					

	
		
def debug():
	# set config 
	config = Config();
	opt = config.flags.FLAGS

	# print(opt.dataset_dir)

	# set sfmlearner
	sfm = SfMLearner(opt);

	#sfm.build_train_graph()

	sfm.train()

debug()

