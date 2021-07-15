import os
import sys
import tensorflow as tf



class Config(object):

	def __init__(self):
		self.flags = tf.app.flags
		self.flags.DEFINE_string("dataset_dir", "/media/mr/file/KITTI_dump/", "Dataset directory")
		self.flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
		self.flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
		self.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
		self.flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
		self.flags.DEFINE_float("smooth_weight", 0.5, "Weight for smoothness")
		self.flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
		self.flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
		self.flags.DEFINE_integer("img_height", 128, "Image height")
		self.flags.DEFINE_integer("img_width", 416, "Image width")
		self.flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
		self.flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
		self.flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
		self.flags.DEFINE_integer("save_latest_freq", 5000, \
			"Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
		self.flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")

		self.flags.DEFINE_integer("num_source", 2, "number source image")
	
		self.flags.DEFINE_integer("num_scales", 4, "number source image")
	

