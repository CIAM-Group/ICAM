import time

import torch

import os
from logging import getLogger

from tqdm import tqdm

from CVRPTWEnv import CVRPTWEnv as Env
from CVRPTWModel_ICAM import CVRPTWModel as Model

from utils.utils import *


class CVRPTWTester:
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

    def run(self):
        self.time_estimator.reset()

        score = AverageMeter()
        aug_score = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        optimal_score = 1.0
        saved_dataset = None
        if self.tester_params['test_data_load']['enable']:
            file_name = self.tester_params['test_data_load']['filename']
            solution_name = self.tester_params['test_data_load']['solution_filename']
            saved_dataset, optimal_score, capacity = self.env.load_dataset(file_name,
                                                                           offset=0,
                                                                           num_samples=test_num_episode,
                                                                           solution_name=solution_name)
            self.logger.info("Data loaded from: {0}".format(file_name))
            self.logger.info('Successfully load {0} CVRPTW{1} instances, optimal: {2:.4f}, capacity: {3:.4f}'.format(
                saved_dataset[1].shape[0], saved_dataset[1].shape[1], optimal_score, capacity))

        episode = 0
        start_time = time.time()
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            depot_xy = saved_dataset[0][episode:episode+batch_size]
            node_xy = saved_dataset[1][episode:episode+batch_size]
            node_demand = saved_dataset[2][episode:episode+batch_size]
            service_time = saved_dataset[3][episode:episode+batch_size]
            tw_start = saved_dataset[4][episode:episode+batch_size]
            tw_end = saved_dataset[5][episode:episode+batch_size]
            batch_dataset = (depot_xy, node_xy, node_demand, service_time, tw_start, tw_end)

            avg_score, avg_aug_score = self._test_one_batch(batch_size,batch_dataset)

            score.update(avg_score, batch_size)
            aug_score.update(avg_aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, avg_score, avg_aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                end_time = time.time()
                self.logger.info(" *** Test Done *** ")
                self.logger.info("===============================================================")
                self.logger.info(" problem size: {0}, pomo size: {1}, distribution: {2}, optimal score: {3:.4f} ".format(
                    self.env_params['problem_size'], self.env_params['pomo_size'], self.env_params['distribution'],optimal_score))
                self.logger.info(" NO-AUG SCORE:{0:.4f}, GAP:{1:.3f}%".format(
                    score.avg, (score.avg - optimal_score) * 100 / optimal_score))
                self.logger.info(" AUGMENTATION SCORE:{0:.4f}, GAP:{1:.3f}%".format(
                    aug_score.avg, (aug_score.avg - optimal_score) * 100 / optimal_score))

                self.logger.info(" Total time: {:.2f} sec".format(end_time - start_time))
                self.logger.info(" Avg time per episode: {:.2f} sec".format((end_time - start_time) / test_num_episode))


    def _test_one_batch(self, batch_size,batch_dataset):

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
            self.env.load_problems_cvrptw(batch_size,
                                          problem_size=self.env_params['problem_size'],
                                          pomo_size=self.env_params['pomo_size'],
                                          validation_data=batch_dataset,
                                          aug_factor=aug_factor,
                                          device=self.device)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            with tqdm(total=0) as pbar:
                while not done:
                    cur_dist = self.env.get_local_feature()
                    selected, _ = self.model(state,cur_dist)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected)
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
