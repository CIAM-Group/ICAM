'''
Remarks:
- This is the test script for CVRP on real-world datasets using ICAM model.

Datasets: we employ the dataset provided by RRNCO (https://github.com/ai4co/real-routing-nco) 
and select eight representative cities 
(Beijing, Chicago, London, Melbourne, Seoul, Sydney, Tokyo, Toronto) across four continents (Asia, Europe, North America, and Oceania)
to ensure diverse urban layout structures. 

We have randomly sampled subgraphs to generate datasets with 100, 200 and 500 customer locations (128 instances per dataset).
Node demands and vehicle capacities are set according to the standard configurations described in LEHD and BQ-NCO papers,
i,e., vehicle capacities are set to 50, 80, and 100 for datasets with 100, 200, and 500 customer locations, respectively.
Near-optimal solutions obtained by the HGS-PyVRP heuristic.

We sincerely thank the authors of RRNCO for sharing the datasets.

Please download the datasets from Google Drive:
https://drive.google.com/drive/folders/1WOpwO65gkhpaxU2VWYExyUdKwOZfX2Hm?usp=sharing

- The dataset files are placed in the folder: ../data/cvrp_real_world/{city_name}/


'''

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

from CVRPTester_RealWorld import CVRPTester as Tester


env_params = {
    'problem_size': None,  # to be set later
    'pomo_size': None,  # to be set later
    'city': None,
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
        'name': 'icam_cvrp', # name of pre-trained model to load
        #'epoch': 1000,  # epoch version of pre-trained model to load.
    },
    'test_episodes': None,  # to be set later
    'test_batch_size': None,  # to be set later
    'augmentation_enable': None,  # to be set later
    'aug_factor': 8,
    'aug_batch_size': None,  # to be set later
    'test_data_load': {
        'enable': True,
        'filename': None,
        'solution_filename': None,
    },
}



##########################################################################################
from datetime import datetime, time
import pytz
process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logger_params = {
    'log_file': {
        'desc': None, # to be set later
        'filename': 'run_log.txt',
        'filepath': None, # to be set later
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
    
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    import re
    logger = logging.getLogger('root')
    ##########################################################################################
    # parameters
    # need to be modified
    augmentation_enable = True
    data_details= { 100: {'test_episodes': 128, 'aug_batch_size': 128},
                    200: {'test_episodes': 128,  'aug_batch_size': 128},
                    500: {'test_episodes': 128,  'aug_batch_size': 128},
                  }
    ##########################################################################################
    
    files_dir = "../data/cvrp_real_world"
    tester_params['augmentation_enable'] = augmentation_enable
    for city in sorted(os.listdir(files_dir)):
        city_dir = os.path.join(files_dir, city)
        if not os.path.isdir(city_dir):
            continue
        for fname in sorted(os.listdir(city_dir)):
            if not fname.endswith(".pt"):
                continue
            pattern = r'(?P<city>[A-Za-z]+)_distance_(?P<nodes>\d+)_cvrp(?P<num>\d+)_capacity(?P<cap>\d+)\.pt'
            match = re.match(pattern, fname)
            if match:
                city = match.group('city')  # Melbourne
                batch_size = int(match.group('nodes'))  # 128
                problem_size = int(match.group('num'))  # 100
                capacity = int(match.group('cap'))
            else:
                continue
            logger.info("================================================================")
            logger.info("================================================================")
            
            env_params['problem_size'] = problem_size
            env_params['pomo_size'] = problem_size
            env_params['city'] = city
            
            tester_params['test_episodes'] = data_details[problem_size]['test_episodes']
            if tester_params['augmentation_enable']:
                tester_params['test_batch_size'] = data_details[problem_size]['aug_batch_size']
                highlight = f'aug{tester_params["aug_factor"]}'
            else:
                tester_params['test_batch_size'] = data_details[problem_size]['test_episodes']
                highlight = 'no_aug'
            
            data_path = os.path.join(city_dir, fname)
            save_sol_pt_name = f"{city}_distance_{batch_size}_cvrp{problem_size}_capacity{capacity}_pyvrp600s.pt"
            save_sol_path = os.path.join(city_dir,save_sol_pt_name)
            
            tester_params['test_data_load']['filename'] = data_path
            tester_params['test_data_load']['solution_filename'] = save_sol_path
            
            logger_params['log_file']['desc'] = f'{highlight}_test_{city}_distance_cvrp{problem_size}_pomo{problem_size}'
            logger_params['log_file']['filepath'] = f'./result_cvrp_test_real_world/{city}/cvrp{problem_size}/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '_'+logger_params['log_file']['desc']

            main()
