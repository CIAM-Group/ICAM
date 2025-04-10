##########################################################################################
# Machine Environment Config

USE_CUDA = True
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
from ATSPTester import ATSPTester as Tester


##########################################################################################
# parameters
# need to be modified
# For atsp, augmentation is not used in the test phase.
# If you want to use, please modify the eval_type in model_params to 'sampling'.
# However, the performance will be worse than the greedy one when facing large-scale instances (e.g., TSP1000), even with aug x128
augmentation_enable = False
problem_size = 1000
pomo_size = problem_size
##########################################################################################
data_details= { 100: {'test_episodes': 10000, 'aug_batch_size': 100,},
                200: {'test_episodes': 128,   'aug_batch_size': 32, },
                500: {'test_episodes': 128,   'aug_batch_size': 8, },
               1000: {'test_episodes': 128,   'aug_batch_size': 2, },
              }
test_episodes = data_details[problem_size]['test_episodes']
test_batch_size = test_episodes
aug_batch_size = data_details[problem_size]['aug_batch_size']

data_path = f'../data/atsp/test_lkh3_atsp{problem_size}_nums{test_episodes}_seed1234_uniform.pt'
solution_path = None

env_params = {
    'problem_size': problem_size,
    'pomo_size': problem_size
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6, # Note that the model has two sub-encoder, each sub-encoder has 6 layers.
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'neighbors': 50, # number of neighbors for each node, used for generating init embeddings
    'eval_type': 'greedy',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../pretrained',  # directory path of pre-trained model and log files saved.
        'name': 'icam_atsp', # name of pre-trained model to load
        #'epoch': 400,  # epoch version of pre-trained model to load.
    },
    'test_data_load': {
        'enable': True,
        'filename': data_path,
        'solution_filename': solution_path,
    },
    'test_episodes': test_episodes,
    'test_batch_size': test_episodes,
    'augmentation_enable': augmentation_enable,
    'aug_factor': 128,
    'aug_batch_size': aug_batch_size,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']
    model_params['eval_type'] = 'sampling'
    highlight = f'aug{tester_params["aug_factor"]}'
else:
    highlight = 'no_aug'

from datetime import datetime
import pytz

process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logger_params = {
    'log_file': {
        'desc': f'{highlight}_test_tsp{problem_size}_pomo{pomo_size}',
        'filename': 'run_log.txt',
        'filepath': './result_tsp_test/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    }
}

##########################################################################################
# main

def main():

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
