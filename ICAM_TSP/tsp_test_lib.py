##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTester_LIB import TSPTester as Tester


##########################################################################################
# parameters
# need to be modified
augmentation_enable = False
detailed_log = True # if True, log for each instance will be output as a list at the end of the log file.
##########################################################################################
lib_path = '../data/TSPLIB_max5000_n70.txt'
env_params = {
    'problem_size': None,
    'pomo_size': None,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 12,
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'eval_type': 'greedy',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../pretrained',  # directory path of pre-trained model and log files saved.
        'name': 'icam_tsp', # name of pre-trained model to load
        #'epoch': 2500,  # epoch version of pre-trained model to load.
    },
    'filename': lib_path,
    'test_episodes': 1,
    'test_batch_size': 1,
    'augmentation_enable': augmentation_enable,
    'aug_factor': 8,
    'aug_batch_size': 1,
    'detailed_log': detailed_log,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']
    highlight = f'aug{tester_params["aug_factor"]}'
else:
    highlight = 'no_aug'

##########################################################################################
from datetime import datetime
import pytz

process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logit_clipping = model_params['logit_clipping']
logger_params = {
    'log_file': {
        'desc': f'{highlight}_test_TSPLIB_max5000_n70',
        'filename': 'run_log.txt',
        'filepath': './result_tsp_test/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run_lib()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
