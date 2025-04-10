
import numpy as np
import torch
from logging import getLogger

from ATSPEnv import ATSPEnv as Env
from ATSPModel_ICAM import ATSPModel as Model

from torch.optim import Adam as Optimizer

from utils.utils import *

from ATSProblemDef import get_saved_data


class ATSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Saved Model Loaded !!')
            self.logger.info("Model loaded from: {0}".format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()

        file_name_100 = f'../data/atsp/test_lkh3_atsp100_nums10000_seed1234_uniform.pt'
        self.saved_problems_100, self.optimal_score_100 = get_saved_data(file_name_100, 10000, self.device)
        self.logger.info('Successfully load {0} TSP{1} instances, optimal: {2:.4f}'.format(
            self.saved_problems_100.shape[0], self.saved_problems_100.shape[1], self.optimal_score_100))

        file_name_1000 = f'../data/atsp/test_lkh3_atsp1000_nums128_seed1234_uniform.pt'
        self.saved_problems_1000, self.optimal_score_1000 = get_saved_data(file_name_1000, 128, self.device)
        self.logger.info('Successfully load {0} TSP{1} instances, optimal: {2:.4f}'.format(
            self.saved_problems_1000.shape[0], self.saved_problems_1000.shape[1], self.optimal_score_1000))

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        self.lr_decay_epoch = self.optimizer_params['lr_decay_epoch']

        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            #########################################################################
            # Train
            #########################################################################
            if epoch == self.lr_decay_epoch:
                self.optimizer.param_groups[0]['lr'] = self.optimizer_params['optimizer']['lr'] * 0.1  # 1e-5
            self.logger.info(' Epoch {:4d}: Current learning rate: {}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            #########################################################################
            # Validation
            #########################################################################
            self._validation_one_epoch(self.saved_problems_100, self.optimal_score_100, epoch)
            self._validation_one_epoch(self.saved_problems_1000, self.optimal_score_1000, epoch)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:4d}/{:4d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))


            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            # save model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score = AverageMeter()
        loss = AverageMeter()

        batches_per_epoch = self.trainer_params['batches_per_epoch']
        loop_cnt = 0  # used for counting the number of batches

        while loop_cnt < batches_per_epoch:
            if epoch <= self.trainer_params['stage1_epochs']:
                true_problem_size = self.env_params['min_problem_size']
                true_batch_size = self.trainer_params['stage1_batch_size']
            else:
                # varying scale of training samples
                true_problem_size = np.random.randint(self.env_params['min_problem_size'],
                                                      self.env_params['max_problem_size'] + 1)
                true_batch_size = int(self.trainer_params['vst_base_batch_size'] * ((100 / true_problem_size) ** 2))

            avg_score, avg_loss = self._train_one_batch(true_problem_size, true_batch_size, epoch)

            score.update(avg_score, true_batch_size)
            loss.update(avg_loss, true_batch_size)

            loop_cnt += 1
            if loop_cnt <= 5 or loop_cnt % 200 == 0:
                self.logger.info('Epoch {:4d}: Train {:4d}/{:4d}({:5.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, loop_cnt, batches_per_epoch, 100. * loop_cnt / batches_per_epoch,
                                             score.avg, loss.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:4d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * loop_cnt / batches_per_epoch,
                                 score.avg, loss.avg))

        return score.avg, loss.avg

    def _train_one_batch(self, problem_size,batch_size,epoch):

        # Prep
        ###############################################
        self.model.train()

        pomo_size = problem_size
        self.env.load_problems_atsp(batch_size,
                                    problem_size=problem_size,
                                    pomo_size=pomo_size,
                                    device=self.device)

        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, pomo_size, 0))
        # shape: (batch, pomo, 0~)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            cur_dist = self.env.get_local_feature()
            selected, prob = self.model(state,cur_dist)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)

        if epoch < self.lr_decay_epoch:
            # In stage 1 and stage 2, we use all trajectories to train the model
            ###############################################
            loss_mean = loss.mean()
        else:
            # In stage 3, we use topk advantage to improve the performance
            ###############################################
            # get topk with the biggest advantage(best reward)
            topk_advantage, topk_advantage_index = torch.topk(advantage, k=self.trainer_params['k_value'],
                                                              dim=-1, largest=True, sorted=True)
            # shape: (batch,k)
            topk_advantage_log_prob = log_prob.gather(dim=-1, index=topk_advantage_index)
            loss_best = -topk_advantage * topk_advantage_log_prob  # Minus Sign: To Increase REWARD
            # shape: (batch, k)
            beta = self.trainer_params['beta']
            loss_mean = loss.mean() + beta * loss_best.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.optimizer.zero_grad()
        loss_mean.backward()
        max_norm = self.trainer_params['max_norm']
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)

        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

    def _validation_one_epoch(self, dataset, optimal_score, epoch):

        self.model.eval()
        self.model.set_decoder_type("greedy")
        batch_size = dataset.size(0)
        problem_size = dataset.size(1)
        with torch.no_grad():
            self.env.load_problems_atsp(batch_size=batch_size,
                                       problem_size=problem_size,
                                       validation_data=dataset,
                                       device=self.device)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                cur_dist = self.env.get_local_feature()
                selected, prob = self.model(state, cur_dist)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        no_aug_score = -max_pomo_reward.float().mean()  # negative sign to make positive value
        gap = ((no_aug_score - optimal_score) * 100 / optimal_score).item()

        avg_score_eval = no_aug_score.item()

        # Logs
        ##################################################
        self.result_log.append(f'eval_{problem_size}', epoch, avg_score_eval)
        self.result_log.append(f'gap_{problem_size}', epoch, gap)

        self.logger.info('Epoch {:4d}: (No aug)In {}-nodes instances, multi-greedy score: {:.4f}, Gap: {:.4f}%'.format(
                epoch, problem_size, avg_score_eval, gap))

        if epoch > 1:
            image_prefix = '{}/latest'.format(self.result_folder)
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=[f'eval_{problem_size}'])
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=[f'gap_{problem_size}'])

        return avg_score_eval, gap