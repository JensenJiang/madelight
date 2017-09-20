import os

import tensorflow as tf


class EvalHelper:
    def __init__(self, sess, config):
        self.sess = sess
        self.modloc = config.modloc
        self.saver = tf.train.Saver()
        self.ckpt_dir = self.modloc.ckpt_dir(config.exp_name)

    def load_checkpoint(self, name):
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, name))
