import time

import numpy as np
import torch

import os
from logging import getLogger

from tqdm import tqdm

from TSPEnv import TSPEnv as Env
from TSPModel_ICAM import TSPModel as Model

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if "epoch" in model_load.keys():
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        else:
            checkpoint_fullname = '{path}/{name}.pt'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info("Model loaded successfully!!!")
        self.logger.info("Model loaded from: {}".format(checkpoint_fullname))

        total = sum([param.nelement() for param in self.model.parameters()])
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # utility
        self.time_estimator = TimeEstimator()

    def run_lib(self):
        self.time_estimator.reset()

        import ast
        result_dict = {}
        result_dict["instances"] = []
        result_dict['optimal'] = []
        result_dict['problem_size'] = []
        result_dict['no_aug_score'] = []
        result_dict['aug_score'] = []
        result_dict['no_aug_gap'] = []
        result_dict['aug_gap'] = []

        gap_set_less_1000 = []
        gap_set_1000_2000 = []
        gap_set_2000_3000 = []
        gap_set_gt_3000 = []

        aug_gap_set_less_1000 = []
        aug_gap_set_1000_2000 = []
        aug_gap_set_2000_3000 = []
        aug_gap_set_gt_3000 = []

        filename = self.tester_params["filename"]
        start_time = time.time()

        with open(filename, 'r') as f:
            for line in f:
                dict_instance_info = {}
                instance = line.strip()  # use strip to remove the '\n' at the end of each line
                instance_list = ast.literal_eval(instance)  # use ast.literal_eval to convert string to list

                name = instance_list[0]  # name of the instance
                optimal = float(instance_list[1])  # optimal value of the instance
                instance_xy = np.array(instance_list[2:]).astype(np.float32)  # shape: (dimension*2,)
                node_coord = torch.from_numpy(instance_xy).reshape(-1, 2).unsqueeze(0)
                # shape: (1,problem_size,2)
                dimension = node_coord.size(1)  # node number of the instance
                assert instance_xy.shape[0] == dimension * 2, "dimension error in instance {}".format(name)

                dict_instance_info['optimal'] = optimal
                dict_instance_info['problem_size'] = dimension
                dict_instance_info['pomo_size'] = dimension
                dict_instance_info['original_node_xy_lib'] = node_coord
                # shape:(1,problem_size,2)
                dict_instance_info['name'] = name

                self.logger.info("===============================================================")
                self.logger.info("Instance name: {0}, problem_size: {1}".format(name, dimension))

                # normalize data to [0,1] using min-max normalization
                ################################################################
                max_value = np.max(instance_xy)
                min_value = np.min(instance_xy)
                nodes_xy_normalized = (node_coord - min_value) / (max_value - min_value)
                dict_instance_info["node_xy"] = nodes_xy_normalized
                # shape:(1,dimension,2)


                score, aug_score = self._test_one_batch_lib(batch_size=1,dict_instance_info=dict_instance_info)

                ############################
                # Logs
                ############################
                no_aug_gap = (score - optimal) * 100 / optimal
                aug_gap = (aug_score - optimal) * 100 / optimal
                result_dict["instances"].append(name)
                result_dict['optimal'].append(optimal)
                result_dict['problem_size'].append(dimension)
                result_dict['no_aug_score'].append(score)
                result_dict['aug_score'].append(aug_score)
                result_dict['no_aug_gap'].append(no_aug_gap)
                result_dict['aug_gap'].append(aug_gap)

                if dimension <= 1000:
                    gap_set_less_1000.append(no_aug_gap)
                    aug_gap_set_less_1000.append(aug_gap)
                elif dimension <= 2000:
                    gap_set_1000_2000.append(no_aug_gap)
                    aug_gap_set_1000_2000.append(aug_gap)
                elif dimension <= 3000:
                    gap_set_2000_3000.append(no_aug_gap)
                    aug_gap_set_2000_3000.append(aug_gap)
                else:
                    gap_set_gt_3000.append(no_aug_gap)
                    aug_gap_set_gt_3000.append(aug_gap)

                self.logger.info("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
                self.logger.info("No aug score:{:.3f}, No aug gap:{:.3f}%".format(score, no_aug_gap))
                self.logger.info("Aug score:{:.3f}, Aug gap:{:.3f}%".format(aug_score, aug_gap))

        end_time = time.time()

        # Logs for all instances
        self.logger.info(" *** Test Done *** ")
        self.logger.info("===============================================================")
        if self.tester_params["detailed_log"]:
            self.logger.info("instance: {0}".format(result_dict['instances']))
            self.logger.info("optimal: {0}".format(result_dict['optimal']))
            self.logger.info("problem_size: {0}".format(result_dict['problem_size']))
            self.logger.info("no_aug_score: {0}".format(result_dict['no_aug_score']))
            self.logger.info("aug_score: {0}".format(result_dict['aug_score']))
            self.logger.info("no_aug_gap: {0}".format(result_dict['no_aug_gap']))
            self.logger.info("aug_gap: {0}".format(result_dict['aug_gap']))
            self.logger.info("===============================================================")

        self.logger.info("size <=1000, number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                         format(len(gap_set_less_1000), np.mean(gap_set_less_1000), np.mean(aug_gap_set_less_1000)))
        self.logger.info("1000<size<=2000, number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(gap_set_1000_2000), np.mean(gap_set_1000_2000), np.mean(aug_gap_set_1000_2000)))
        self.logger.info("2000<size<=3000, number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(gap_set_2000_3000), np.mean(gap_set_2000_3000), np.mean(aug_gap_set_2000_3000)))
        self.logger.info("size>3000, number: {0}, avg gap(no aug): {1:.3f}%, avg gap(aug): {2:.3f}%".
                            format(len(gap_set_gt_3000), np.mean(gap_set_gt_3000), np.mean(aug_gap_set_gt_3000)))

        self.logger.info("===============================================================")
        avg_all_no_aug_gap = np.mean(result_dict['no_aug_gap'])  # avg of all instances gap (no aug)
        avg_all_aug_gap = np.mean(result_dict['aug_gap'])  # avg of all instances gap (aug)
        all_instance_num = len(result_dict['instances'])
        max_dimension = max(result_dict['problem_size'])
        min_dimension = min(result_dict['problem_size'])
        self.logger.info("All instances number: {0}, min_dimension: {1}, max_dimension: {2}, avg gap(no aug): {3:.3f}%, avg_gap(aug): {4:.3f}%".
                         format(all_instance_num, min_dimension, max_dimension, avg_all_no_aug_gap, avg_all_aug_gap))
        self.logger.info("Avg time per instance: {0:.2f}s".format((end_time - start_time) / all_instance_num))


    def _test_one_batch_lib(self, batch_size,dict_instance_info):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            problem_size = dict_instance_info['problem_size']
            self.env.load_problems_tsp(batch_size, problem_size,lib_data=dict_instance_info,
                                       aug_factor=aug_factor,device=self.device)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            with tqdm(total=0) as pbar:
                while not done:
                    cur_dist = self.env.get_local_feature()
                    selected, prob = self.model(state, cur_dist)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected,lib_mode=True)
                    pbar.total += 1
                    pbar.update(1)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()
