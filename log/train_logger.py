import os
import sys
from log.basic_logger import BasicLogger
import time

if sys.path[-1] != os.getcwd():
    sys.path.append(os.getcwd())


def create_dir(dir_list):
    assert isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)


class TrainLogger(BasicLogger):
    def __init__(self, args, create=True):
        self.args = args

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        savetag = timestamp + '_' + args.model
        save_dir = args.save_dir
        train_save_dir = os.path.join(save_dir, savetag)
        self.log_dir = os.path.join(train_save_dir, 'log', 'train')
        self.model_dir = os.path.join(train_save_dir, 'model')

        if create:
            create_dir([self.log_dir, self.model_dir])
            print(self.log_dir)
            log_path = os.path.join(self.log_dir, 'Train.log')
            super().__init__(log_path)

    def get_log_dir(self):
        if hasattr(self, 'log_dir'):
            return self.log_dir
        else:
            return None

    def get_model_dir(self):
        if hasattr(self, 'model_dir'):
            return self.model_dir
        else:
            return None
