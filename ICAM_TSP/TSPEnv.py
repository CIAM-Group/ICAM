import math
from dataclasses import dataclass

import torch

from TSProblemDef import get_random_problems_tsp, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
    dist: torch.Tensor
    # shape: (batch, problem, problem)
    log_scale: float


@dataclass
class Step_State:
    batch_size: torch.Tensor
    pomo_size: torch.Tensor
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = None
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
        # shape: (batch, pomo, 0~problem)

        self.dist = None

        self.FLAG__use_saved_problems = False
        self.device = None
        self.saved_node_xy = None
        self.saved_index = None

        self.original_node_xy_lib = None # for lib data
        self.optimal = None

    def input_saved_data(self,problems,device):
        self.FLAG__use_saved_problems = True
        self.saved_node_xy = problems
        self.saved_index = 0
        self.device = device

    def load_problems_tsp(self, batch_size, problem_size,pomo_size=None,lib_data=None, validation_data=None, aug_factor=1,device=None):
        self.batch_size = batch_size
        self.problem_size = problem_size
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if device is not None:
            self.device = device

        if lib_data is not None:
            self.problems = lib_data["node_xy"].to(device)
            # shape: (1, problem_size, 2)
            self.original_node_xy_lib = lib_data['original_node_xy_lib'].to(self.device)
            # shape: (1, problem_size, 2)
        elif validation_data is not None:
            self.problems = validation_data.to(self.device)
            # shape: (batch, problem_size, 2)
        else:
            if not self.FLAG__use_saved_problems:
                self.problems = get_random_problems_tsp(batch_size, self.problem_size)
                # problems.shape: (batch, problem, 2)
            else:
                self.problems = self.saved_node_xy[self.saved_index:self.saved_index + batch_size].to(self.device)
                # shape: (batch, problem_size, 2)
                self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError(f'The augmentation factor {aug_factor} is not implemented.')

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)

        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)
        # CREATE STEP STATE
        self.step_state = Step_State(batch_size=self.batch_size,pomo_size=self.pomo_size)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        # Note that for "torch.cdist" function, compute_mode must be 'donot_use_mm_for_euclid_dist'.
        # Because we find if compute_mode is other mode, such as 'use_mm_for_euclid_dist' and 'use_mm_for_euclid_dist_if_necessary', the obtained matrix is wrong.
        # It is weird, and we have no idea why.
        # The official document(https://pytorch.org/docs/2.4/generated/torch.cdist.html) does not mention this issue.
        # We also provide a double check code to verify the correctness of the distance matrix.
        self.dist = torch.cdist(self.problems, self.problems, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        # shape: (batch, problem, problem)
        #dist_double_check = (self.problems[:, :, None, :] - self.problems[:, None, :, :]).norm(p=2, dim=-1)
        #assert (self.dist == dist_double_check).all(), \
        #    'The distance matrix is wrong due to the precision problem, please check your compute_mode in cdist function.'

        log_scale = math.log2(self.problem_size)

        self.reset_state = Reset_State(problems=self.problems,dist=self.dist,log_scale=log_scale)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected,lib_mode=False):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node.unsqueeze(-1)), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask.scatter_(dim=-1, index=self.current_node.unsqueeze(-1), value=float('-inf'))
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)

        if done:
            # judge whether solution is valid.
            assert (self.step_state.ninf_mask == float('-inf')).all(), \
                            'ninf_mask is not all -inf, but done is True, so the solution is not valid.'
            reward = -self._get_travel_distance(lib_mode)  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self,lib_mode):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        if not lib_mode:
            seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        else:
            assert self.original_node_xy_lib.size(0) == 1, 'The original_node_xy_lib should be a single instance.'
            self.original_node_xy_lib = self.original_node_xy_lib.expand(self.batch_size, self.problem_size,2) # batch size is 1 or 8 (aug)
            seq_expanded = self.original_node_xy_lib[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def get_local_feature(self):
        if self.current_node is None:
            return None

        current_node = self.current_node.unsqueeze(-1).expand(self.batch_size, self.pomo_size, self.problem_size)
        # shape: (batch, pomo, problem)
        cur_dist = self.dist.gather(dim=1, index=current_node)
        # shape: (batch, pomo, problem)

        return cur_dist


