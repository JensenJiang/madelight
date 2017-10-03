import os
import pickle

import tensorflow as tf

from ..utils.logger import TensorBoardLogger


class TrainClock:
    def __init__(self):
        self.global_step = 0
        self.cur_epoch = 0
        self.cur_epoch_step = 0

    def tick(self):
        self.global_step += 1
        self.cur_epoch_step += 1

    def tock(self):
        self.cur_epoch += 1
        self.cur_epoch_step = 0


class TrainHelper:
    def __init__(self, sess, config):
        # Model locator
        self.modloc = config.modloc

        # Training related paths
        self.train_log_path = self.modloc.exp_train_log_dir(config.exp_name)
        self.ckpt_dir = self.modloc.ckpt_dir(config.exp_name)

        # Training resources
        self.sess = sess
        self.saver = tf.train.Saver(max_to_keep=None)  # Save all the checkpoints.
        self.clock = TrainClock()

        # Initialization
        self._init_log_dirs()

        # Dump config file and print
        config.save_config_default_path()
        config.print_config()

    def _init_log_dirs(self):
        # Models path
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def save_checkpoint(self, name):
        ckpt_path = os.path.join(self.ckpt_dir, name)
        clock_path = os.path.join(self.ckpt_dir, name + r'.clock')
        self.saver.save(self.sess, ckpt_path)
        with open(clock_path, 'wb') as fout:
            pickle.dump(self.clock, fout)

    def load_checkpoint(self, ckpt_path):
        clock_path = ckpt_path + r'.clock'
        self.saver.restore(self.sess, ckpt_path)
        with open(clock_path, 'rb') as fin:
            self.clock = pickle.load(fin)

    def create_tbs(self, *tb_names):
        self.tbs = [TensorBoardLogger(os.path.join(self.train_log_path, name)) for name in tb_names]
        return self.tbs
