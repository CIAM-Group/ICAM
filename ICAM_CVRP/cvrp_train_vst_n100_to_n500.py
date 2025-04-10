##########################################################################################
# Machine Environment Config
import random

import numpy as np
import torch

USE_CUDA = True
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "../")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTrainer import CVRPTrainer as Trainer


##########################################################################################
# parameters which should be set before running
min_problem_size = 100 # minimum size for training
max_problem_size = 500 # maximum size for training

min_capacity = 50 # minimum capacity for training
max_capacity = 100 # maximum capacity for training

training_epochs = 1000 # total epochs for training
batches_per_epoch = 1000 # steps per epoch
stage1_epochs = 100 # epochs for stage 1
stage3_epochs = 200 # epochs for stage 3
stage1_batch_size = 128 # batch size for training, only used in stage 1
vst_base_batch_size = 128  # It can be adjusted based on the GPU memory used
##########################################################################################
assert max_problem_size >= min_problem_size
assert stage1_epochs + stage3_epochs <= training_epochs

env_params = {
    'min_problem_size': min_problem_size,
    'max_problem_size': max_problem_size,

    'min_capacity': min_capacity,
    'max_capacity': max_capacity,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 12,
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'eval_type': 'sampling',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
    },
    'lr_decay_epoch': training_epochs - stage3_epochs + 1,
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': training_epochs,
    'stage1_epochs': stage1_epochs,
    'batches_per_epoch': batches_per_epoch,
    'stage1_batch_size': stage1_batch_size,
    'vst_base_batch_size': vst_base_batch_size,
    'logging': {
        'model_save_interval': 1,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_score.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        #'path': '',  # directory path of pre-trained model and log files saved.
        #'epoch': ,  # epoch version of pre-trained model to load.

    },
    'max_norm': 5.0,
    'k_value': 20,  # used for stage 3
    'beta': 0.1,  # used for stage 3
}

import pytz
from datetime import datetime
process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))

logger_params = {
    'log_file': {
        'desc': f"icam_cvrp{min_problem_size}_to_{max_problem_size}_epoch{training_epochs}",
        'filename': 'run_log.txt',
        'filepath': './result_cvrp_models/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}


##########################################################################################
# main
def main():
    create_logger(**logger_params)
    _print_config()
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)
    copy_all_src(trainer.result_folder)
    trainer.run()


def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
##########################################################################################

if __name__ == "__main__":
    seed_everything()
    main()
