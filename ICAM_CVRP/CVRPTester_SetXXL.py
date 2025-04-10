import time

import numpy as np
import torch

import os
from logging import getLogger

from tqdm import tqdm

from CVRPEnv import CVRPEnv as Env
from CVRPModel_ICAM import CVRPModel as Model

from utils.utils import *


class CVRPTester:
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
        result_dict['time'] = []

        start_time = time.time()
        filename = self.tester_params["filename"]
        with open(filename, 'r') as f:
            for line in f:
                instance_start_time = time.time()
                dict_instance_info = {}
                instance = line.strip()  # use strip to remove the '\n' at the end of each line
                instance_list = ast.literal_eval(instance)  # use ast.literal_eval to convert string to list

                name = instance_list[1]  # name of the instance
                optimal = instance_list[-2]  # optimal value of the instance

                depot_index = instance_list.index('depot')  # index of the depot
                depot_xy = instance_list[depot_index + 1:depot_index + 3]  # depot location
                customer_index = instance_list.index('customer')  # index of the customer
                demand_index = instance_list.index('demand')  # index of the demand
                capacity_index = instance_list.index('capacity')  # index of the capacity
                customer_xy = instance_list[customer_index + 1:demand_index]  # customer location list,shape:(dimension*2,)
                dimension = len(customer_xy) // 2  # node number of the instance,not including the depot

                if dimension > 7000:
                    continue
                demands = instance_list[demand_index + 1:capacity_index]  # demand,shape:(dimension+1,)
                capacity = instance_list[capacity_index + 1]  # capacity,shape:(1,)

                instance_xy = np.array(depot_xy + customer_xy)  # shape:(dimension*2+2,)
                node_coord = torch.from_numpy(instance_xy.astype(np.float32)).reshape(-1, 2).unsqueeze(0)
                # shape:(1,dimension+1,2)
                demand_normalized = torch.from_numpy(np.array(demands).astype(np.float32)) / float(capacity) # shape:(dimension+1,)
                dict_instance_info['original_depot_node_xy_lib'] = node_coord
                dict_instance_info['node_demand'] = demand_normalized[1:][None, :]  # not including the depot node
                # shape:(1,problem_size)
                dict_instance_info['optimal'] = optimal
                dict_instance_info['problem_size'] = dimension
                dict_instance_info['pomo_size'] = dimension


                min_x = torch.min(node_coord[:, :, 0], dim=-1)[0][:, None]
                min_y = torch.min(node_coord[:, :, 1], dim=-1)[0][:, None]
                max_x = torch.max(node_coord[:, :, 0], dim=-1)[0][:, None]
                max_y = torch.max(node_coord[:, :, 1], dim=-1)[0][:, None]
                # shape:(1,1)
                depot_node_x_normalized = (node_coord[:, :, 0] - min_x) / (max_x - min_x)  # shape:(1,dimension+1)
                depot_node_y_normalized = (node_coord[:, :, 1] - min_y) / (max_y - min_y)  # shape:(1,dimension+1)

                nodes_xy_normalized = torch.cat((depot_node_x_normalized[:, :, None], depot_node_y_normalized[:, :, None]),dim=2)
                # shape:(1,dimension+1,2)
                dict_instance_info["depot_xy"] = nodes_xy_normalized[0, 0][None, None, :]  # # shape: (1, 1, 2)
                dict_instance_info["node_xy"] = nodes_xy_normalized[0, 1:][None, :]  # shape: (1, problem, 2)

                self.logger.info("===============================================================")
                self.logger.info("Instance name: {0}, problem_size: {1}".format(name, dimension))

                score, aug_score = self._test_one_batch(batch_size=1,dict_instance_info=dict_instance_info)

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
                instance_end_time = time.time()
                result_dict['time'].append(instance_end_time - instance_start_time)

                self.logger.info("Instance name: {}, optimal score: {:.4f}".format(name, optimal))
                self.logger.info("No aug score:{:.3f}, No aug gap:{:.3f}%".format(score, no_aug_gap))
                self.logger.info("Aug score:{:.3f}, Aug gap:{:.3f}%".format(aug_score, aug_gap))
                self.logger.info("Instance time: {:.2f}s".format(instance_end_time - instance_start_time))

        end_time = time.time()
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

        avg_all_no_aug_gap = np.mean(result_dict['no_aug_gap'])  # avg of all instances gap (no aug)
        avg_all_aug_gap = np.mean(result_dict['aug_gap'])  # avg of all instances gap (aug)
        all_instance_num = len(result_dict['instances'])
        max_dimension = max(result_dict['problem_size'])
        min_dimension = min(result_dict['problem_size'])
        self.logger.info(
            "All instances number: {0}, min_dimension: {1}, max_dimension: {2}, avg gap(no aug): {3:.3f}%, avg gap(aug): {4:.3f}%".
            format(all_instance_num, min_dimension, max_dimension, avg_all_no_aug_gap, avg_all_aug_gap))
        self.logger.info("Avg time per instance: {0:.2f}s".format((end_time - start_time) / all_instance_num))

    def _test_one_batch(self, batch_size, dict_instance_info):

        # Augmentation
        ###############################################
        problem_size = dict_instance_info['problem_size']
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            pomo_size = min(problem_size, 5000) # due to memory issue, we limit the pomo size to 5000 when adopting augmentation
        else:
            aug_factor = 1
            pomo_size = problem_size

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():

            self.env.load_problems_cvrp(batch_size, problem_size, pomo_size=pomo_size,
                                        lib_data=dict_instance_info,
                                        aug_factor=aug_factor, device=self.device)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            with tqdm(total=0) as pbar:
                while not done:
                    cur_dist = self.env.get_local_feature()
                    selected, _ = self.model(state, cur_dist)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected, lib_mode=True)
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
