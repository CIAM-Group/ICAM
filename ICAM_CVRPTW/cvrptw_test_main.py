##########################################################################################
# Machine Environment Config
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

from CVRPTWTester import CVRPTWTester as Tester


##########################################################################################
# parameters
# need to be modified
augmentation_enable = True
problem_size = 100
pomo_size = problem_size
distribution = "uniform" # "uniform", "rotation" or "explosion"
if pomo_size == 1:
    augmentation_enable = False
##########################################################################################

data_details= { 100: {'test_episodes': 10000, 'aug_batch_size': 1000, "capacity":50},
                200: {'test_episodes': 128,   'aug_batch_size': 128, "capacity": 80},
                500: {'test_episodes': 128,   'aug_batch_size': 128, "capacity": 100},
               1000: {'test_episodes': 128,   'aug_batch_size': 32, "capacity": 250},
              }
test_episodes = data_details[problem_size]['test_episodes']
test_batch_size = test_episodes
aug_batch_size = data_details[problem_size]['aug_batch_size']
capacity = data_details[problem_size]['capacity']

data_path = f'../data/cvrptw/vrptw{problem_size}_uniform_capacity{capacity}_seed1234.pkl'
solution_path = f"../data/cvrptw/hgs_pyvrp_vrptw{problem_size}_uniform_capacity{capacity}_seed1234.pkl"

env_params = {
    'problem_size': problem_size,
    'pomo_size': pomo_size,
    'distribution': distribution,
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
        'name': 'icam_cvrptw', # name of pre-trained model to load
        #'epoch': 1000,  # epoch version of pre-trained model to load.
    },
    'test_episodes': test_episodes,
    'test_batch_size': test_batch_size,
    'augmentation_enable': augmentation_enable,
    'aug_factor': 8,
    'aug_batch_size': aug_batch_size,
    'test_data_load': {
        'enable': True,
        'filename': data_path,
        'solution_filename': solution_path,
    },
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
logger_params = {
    'log_file': {
        'desc': f'{highlight}_test_{distribution}_cvrptw{problem_size}_pomo{pomo_size}',
        'filename': 'run_log.txt',
        'filepath': f'./result_cvrptw_test/cvrptw{problem_size}/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
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
