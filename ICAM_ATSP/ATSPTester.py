
import time

import torch

import os
from logging import getLogger

from tqdm import tqdm

from ATSPEnv import ATSPEnv as Env
from ATSPModel_ICAM import ATSPModel as Model

from utils.utils import *
from ATSProblemDef import get_saved_data

class ATSPTester:
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

    def run(self):

        self.time_estimator.reset()
        start_time = time.time()
        score = AverageMeter()
        aug_score = AverageMeter()
        single_greedy_score = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        optimal_score = 1.0

        if self.tester_params['test_data_load']['enable']:
            file_name = self.tester_params['test_data_load']['filename']
            solution_name = self.tester_params['test_data_load']['solution_filename']

            problems, optimal_score = get_saved_data(file_name,test_num_episode,self.device,start=0,solution_name=solution_name)

            self.env.input_saved_data(problems, self.device)
            self.logger.info("Saved dataset loaded successfully!!!")
            self.logger.info("Data loaded from: {}".format(file_name))
            self.logger.info('problem_size: {0} ,optimal: {1:.4f}'.format(problems.shape[1], optimal_score))

        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            avg_score, avg_aug_score,avg_single_greedy_score = self._test_one_batch(batch_size)

            single_greedy_score.update(avg_single_greedy_score, batch_size)
            score.update(avg_score, batch_size)
            aug_score.update(avg_aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], single_greedy_score:{:.3f}, score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, avg_single_greedy_score, avg_score, avg_aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                end_time = time.time()
                self.logger.info(" *** Test Done *** ")
                self.logger.info("===============================================================")
                self.logger.info(" problem size: {0}, pomo size: {1}, optimal score: {2:.4f} ".format(
                    self.env_params['problem_size'], self.env_params['pomo_size'], optimal_score))
                self.logger.info(" NO-AUG SINGLE GREEDY SCORE:{0:.4f}, GAP:{1:.3f}%".format(
                    single_greedy_score.avg, (single_greedy_score.avg - optimal_score) * 100 / optimal_score))
                self.logger.info(" NO-AUG SCORE:{0:.4f}, GAP:{1:.3f}%".format(
                    score.avg, (score.avg - optimal_score) * 100 / optimal_score))
                self.logger.info(" AUGMENTATION SCORE:{0:.4f}, GAP:{1:.3f}%".format(
                    aug_score.avg, (aug_score.avg - optimal_score) * 100 / optimal_score))

                self.logger.info(" Total time: {:.2f} sec".format(end_time - start_time))
                self.logger.info(" Avg time per episode: {:.2f} sec".format((end_time - start_time) / test_num_episode))


    def _test_one_batch(self, batch_size):


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
            self.env.load_problems_atsp(batch_size,
                                        problem_size=self.env_params['problem_size'],
                                        pomo_size=self.env_params['pomo_size'],
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
            single_greedy_score = -aug_reward[0, :, 0].float().mean()  # don't use multi-greedy

            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

            return no_aug_score.item(), aug_score.item(), single_greedy_score.item()
