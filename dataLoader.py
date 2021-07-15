import os
import sys
import tensorflow as tf
import random
import numpy


class DataLoader(object):
	def __init__(self, 
				 dataset_dir,
				 batch_size,
				 img_height,
				 img_width,
				 seq_length,
				 num_source,
				 num_scales):
		self.dataset_dir = dataset_dir
		self.batch_size = batch_size
		self.img_height = img_height
		self.img_width = img_width
		self.seq_length = seq_length
		self.num_source = num_source
		self.num_scales = num_scales

	def format_file_list(self, dataset_dir, split):
		with open(dataset_dir + split, 'r') as f:
			frames = f.readlines()

		subfolders = [x.split()[0] for x in frames]
		frame_ids = [x.split()[1] for x in frames]	

		image_file_list = [os.path.join(dataset_dir, subfolders[i],
										frame_ids[i] + '.jpg') 
										for i in range(len(subfolders))]

		cam_file_list = [os.path.join(dataset_dir, subfolders[i], 
									  frame_ids[i] + '_cam.txt')
									  for i in range(len(subfolders))]
		"""
		print(len(frame_ids))
		print(len(cam_file_list))
		for image_file in image_file_list:
			print(image_file)
		for cam_file in cam_file_list:
			print(cam_file)
		"""

		all_list = {}
		all_list["image_file_list"] = image_file_list
		all_list["cam_file_list"] = cam_file_list
		return all_list

	
	def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
		tgt_start_idx = int(img_width * (num_source // 2))
		tgt_image = tf.slice(image_seq,
							 [0, tgt_start_idx, 0],
							 [-1, img_width, -1])

		# source image before tgt_image
		src_image_1 = tf.slice(image_seq,
							   [0, 0, 0],
							   [-1, int(img_width * (num_source // 2)), -1])

		# source image after tgt_image
		src_image_2 = tf.slice(image_seq,
							   [0, int(tgt_start_idx + img_width), 0],
							   [-1, int(img_width * (num_source // 2)), -1])

		src_image_seq = tf.concat([src_image_1, src_image_2], axis = 1)

		# src_image_stack [H, W, N * 3] (H is height, W is width, N is num_source)
		src_image_stack = tf.concat([tf.slice(src_image_seq,
											  [0, i * img_width, 0],
											  [-1, img_width, -1])
											  for i in range (num_source)], axis = 2)
		src_image_stack.set_shape([img_height,
								   img_width,
								   num_source * 3])

		tgt_image.set_shape([img_height, img_width, 3])
		return tgt_image, src_image_stack

	def make_intrinsics_matrix(self, fx, fy, cx, cy):
		# Assumes batch input
		batch_size = fx.get_shape().as_list()[0]
		zeros = tf.zeros_like(fx)
		r1 = tf.stack([fx, zeros, cx], axis = 1)
		r2 = tf.stack([zeros, fy, cy], axis = 1)
		r3 = tf.constant([0., 0., 1.], shape=[1, 3])
		r3 = tf.tile(r3, [batch_size, 1])
		intrinsics = tf.stack([r1, r2, r3], axis=1)	
		return intrinsics

	def data_augmentation(self, image, intrinsics, out_h, out_w):
		# Random scaling
		def random_scaling(image, intrinsics):
			batch_size, in_h, in_w, _ = image.get_shape().as_list()
			scaling = tf.random_uniform([2], 1, 1.15)
			x_scaling = scaling[0]
			y_scaling = scaling[1]
			out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
			out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
			image = tf.image.resize_area(image, [out_h, out_w])
			fx = intrinsics[:, 0, 0] * x_scaling
			fy = intrinsics[:, 1, 1] * y_scaling
			cx = intrinsics[:, 0, 2] * x_scaling
			cy = intrinsics[:, 1, 2] * y_scaling
			intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
			return image, intrinsics

		# Random cropping
		def random_cropping(image, intrinsics, out_h, out_w):
			# batch_size, in_h, in_w, _ = image.get_shape().as_list()
			batch_size, in_h, in_w, _ = tf.unstack(tf.shape(image))
			offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype = tf.int32)[0]
			offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype = tf.int32)[0]
			image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, out_h, out_w)

			fx = intrinsics[:, 0, 0]
			fy = intrinsics[:, 1, 1]
			cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype = tf.float32)
			cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype = tf.float32)
			intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
			return image, intrinsics
		image, intrinsics = random_scaling(image, intrinsics)
		image, intrinsics = random_cropping(image, intrinsics, out_h, out_w)
		image = tf.cast(image, dtype = tf.uint8)
		return image, intrinsics


	def get_multi_scale_intrinsics(self, intrinsics, num_scales):
		intrinsics_mscale = []
		for s in range(num_scales):
			fx = intrinsics[:, 0, 0] / ( 2 ** s)
			fy = intrinsics[:, 1, 1] / ( 2 ** s)
			cx = intrinsics[:, 0, 2] / ( 2 ** s)
			cy = intrinsics[:, 1, 2] / ( 2 ** s)
			intrinsics_mscale.append(
				self.make_intrinsics_matrix(fx, fy, cx, cy))

		#print("intrinsics_mscale shape before stack")
		#print(len(intrinsics_mscale))
		intrinsics_mscale = tf.stack(intrinsics_mscale, axis = 1)
		return intrinsics_mscale



	def load_train_batch(self):
		"""
		load a batch of training instances
		"""
		seed = random.randint(0, 2**31 - 1)
		# load the list of training files into queues
		file_list = self.format_file_list(self.dataset_dir, "train.txt")

		#print(file_list)

		# 通过tf.train.string_input_producer() 创建文件名队列
		image_paths_queue = tf.train.string_input_producer(
			file_list["image_file_list"],
			seed = seed,
			shuffle = True)

		cam_paths_queue = tf.train.string_input_producer(
			file_list["cam_file_list"],
			seed = seed,
			shuffle = True)

		self.steps_per_epoch = int(len(file_list['image_file_list']) // self.batch_size)

		# load images
		# 通过tf.WholeFileReader().read(), 从文件名队列中读取文件 
		img_readers = tf.WholeFileReader()
		_, image_contents = img_readers.read(image_paths_queue)
		image_seq = tf.image.decode_jpeg(image_contents)

		tgt_image, src_image_stack = \
			self.unpack_image_sequence(image_seq, self.img_height, self.img_width, self.num_source)

		"""
		image_seq.set_shape([self.img_height, self.img_width * self.seq_length, 3])

		image_seqs = tf.train.batch([image_seq], batch_size = self.batch_size)
		"""

		# load camera intrinsics
		cam_readers = tf.TextLineReader()
		_, raw_cam_contents = cam_readers.read(cam_paths_queue)
		rec_def = []
		for i in range(9):
			rec_def.append([1.])
		raw_cam_vec = tf.decode_csv(raw_cam_contents, record_defaults = rec_def)
		raw_cam_vec = tf.stack(raw_cam_vec)
		intrinsics = tf.reshape(raw_cam_vec, [3, 3])

		# Form training batches
		src_image_stack, tgt_image, intrinsics = \
			tf.train.batch([src_image_stack, tgt_image, intrinsics],
							batch_size = self.batch_size)

		# Data augmentation
		# image_all : B * H * W * (N + 1) * 3, which(B is batch_size, + 1 is tgt image)
		image_all = tf.concat([tgt_image, src_image_stack], axis = 3)

		image_all, intrinsics = self.data_augmentation(
			image_all, intrinsics, self.img_height, self.img_width)
		
		"""
		auged_tgt_image = tf.slice(image_all,
									[0, 0, 0, 0],
									[-1, -1, -1, 3])
		"""
		tgt_image = image_all[:, :, :, :3]
		src_image_stack = image_all[:, :, :, 3:]
		# intrinsics : B * S * 3 * 3, which(B is batch_size, S is num_scales)
		intrinsics = self.get_multi_scale_intrinsics(
			intrinsics, self.num_scales)

		return intrinsics, tgt_image, src_image_stack#, auged_tgt_image

