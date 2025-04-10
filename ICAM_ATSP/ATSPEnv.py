

from dataclasses import dataclass

import math
import torch

from ATSProblemDef import get_random_problems_atsp


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, node, node)
    log_scale: float


@dataclass
class Step_State:
    batch_size: torch.Tensor
    pomo_size: torch.Tensor
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class ATSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.node_cnt = None
        self.pomo_size = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # STEP-State
        ####################################
        self.step_state = None
        self.device = None
        self.FLAG__use_saved_problems = False
        self.saved_index = None
        self.saved_problems = None
        self.optimal = None

    def input_saved_data(self, problems, device):
        self.FLAG__use_saved_problems = True
        self.saved_problems = problems
        self.saved_index = 0
        self.device = device

    def load_problems_atsp(self, batch_size,problem_size,pomo_size=None, validation_data=None, aug_factor=1,device=None):
        self.batch_size = batch_size
        self.node_cnt = problem_size
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if device is not None:
            self.device = device
        if validation_data is not None:
            self.problems = validation_data.to(self.device)
        else:
            if not self.FLAG__use_saved_problems:
                problem_gen_params = self.env_params['problem_gen_params']
                self.problems = get_random_problems_atsp(batch_size, self.node_cnt, problem_gen_params)
                # shape: (batch, node, node)
            else:
                self.problems = self.saved_problems[self.saved_index:self.saved_index + self.batch_size].to(self.device)
                self.saved_index += self.batch_size

        if aug_factor > 1:
            self.problems = self.problems.repeat(aug_factor, 1, 1)
            self.batch_size = self.batch_size * aug_factor


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)
        log_scale = math.log2(self.node_cnt)

        self.step_state = Step_State(batch_size=self.batch_size, pomo_size=self.pomo_size)
        self.step_state.current_node = None  # None
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.node_cnt))
        # shape: (batch, pomo, node)

        reward = None
        done = False
        return Reset_State(problems=self.problems,log_scale=log_scale), reward, done


    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, node_idx):
        # node_idx.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = node_idx
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node.unsqueeze(-1)), dim=2)
        # shape: (batch, pomo, 0~node)

        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask.scatter_(dim=-1, index=self.current_node.unsqueeze(-1), value=float('-inf'))
        # shape: (batch, pomo, node)
        
        # returning values
        done = (self.selected_count == self.node_cnt)
        if done:
            # judge whether solution is valid.
            assert (self.step_state.ninf_mask == float('-inf')).all(), \
                'ninf_mask is not all -inf, but done is True, so the solution is not valid.'
            reward = -self._get_total_distance()  # Note the MINUS Sign ==> We MAXIMIZE reward
            # shape: (batch, pomo)
        else:    
            reward = None
        return self.step_state, reward, done


    def _get_total_distance(self):

        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        batch_index = BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.node_cnt)
        # shape: (batch, pomo, node)
        selected_cost = self.problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance

    def get_local_feature(self):
        if self.current_node is None:
            return None

        current_node = self.current_node.unsqueeze(-1).expand(self.batch_size, self.pomo_size, self.node_cnt)
        # shape: (batch, pomo, problem)
        cur_dist = self.problems.gather(dim=1, index=current_node)
        # shape: (batch, pomo, problem)

        return cur_dist
