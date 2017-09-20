import os


class ModelLocator:
    def __init__(self, model_dir=None):
        self.model_dir = os.path.curdir if model_dir is None else model_dir

    @property
    def root_train_log_dir(self):
        return os.path.join(self.model_dir, r'train_log')

    @property
    def exp_config_dir(self):
        return os.path.join(self.model_dir, r'experiment')

    def exp_train_log_dir(self, exp_name):
        return os.path.join(self.root_train_log_dir, exp_name)

    def ckpt_dir(self, exp_name):
        return os.path.join(self.exp_train_log_dir(exp_name), r'ckpt')
