from dataclasses import dataclass

import math
import torch

from CVRProblemDef import get_random_problems_cvrp, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    dist: torch.Tensor = None
    # shape: (batch, problem, problem)
    log_scale: float = None


@dataclass
class Step_State:
    batch_size: torch.Tensor = None
    pomo_size: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = None
        self.pomo_size = None

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None
        self.device = None

        self.original_depot_node_xy_lib = None # for lib data

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.dist = None # shape: (batch, problem+1, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        self.round_error_epsilon = 0.00001 # for precision stability

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def input_saved_data(self,depot_xy, node_xy, node_demand,device):
        self.FLAG__use_saved_problems = True
        self.saved_depot_xy = depot_xy
        self.saved_node_xy = node_xy
        self.saved_node_demand = node_demand
        self.saved_index = 0
        self.device = device

    def load_problems_cvrp(self, batch_size, problem_size,capacity=None,pomo_size=None,lib_data=None, validation_data=None, aug_factor=1, device=None):
        self.batch_size = batch_size
        self.problem_size = problem_size
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if device is not None:
            self.device = device

        if lib_data is not None:
            depot_xy = lib_data["depot_xy"].to(device)  # # shape: (1, 1, 2)
            node_xy = lib_data["node_xy"].to(device)  # shape: (1, problem, 2)
            node_demand = lib_data['node_demand'].to(device)  # not including the depot node
            self.original_depot_node_xy_lib = lib_data['original_depot_node_xy_lib'].to(device)  # shape: (1, problem+1, 2)
        elif validation_data is not None:
            depot_xy = validation_data["depot_xy"].to(device)
            # shape: (batch, 1, 2)
            node_xy = validation_data["node_xy"].to(device)
            # shape: (batch, problem, 2)
            node_demand = validation_data["node_demand"].to(device)
            # shape: (batch, problem)
        else:
            if not self.FLAG__use_saved_problems:
                assert capacity is not None, "capacity must be given when generating random problems."
                depot_xy, node_xy, node_demand = get_random_problems_cvrp(batch_size, self.problem_size, capacity)
            else:
                depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
                node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
                node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
                self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError(f'The augmentation factor {aug_factor} is not implemented.')

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        # Note that for "torch.cdist" function, compute_mode must be 'donot_use_mm_for_euclid_dist'.
        # For more details about this issue, please refer to the TSPEnv.py file.
        self.dist = torch.cdist(self.depot_node_xy, self.depot_node_xy, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        self.reset_state.dist = self.dist
        # shape: (batch, problem+1, problem+1)
        self.reset_state.log_scale = math.log2(self.problem_size)

        self.step_state.batch_size = self.batch_size
        self.step_state.pomo_size = self.pomo_size

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected,lib_mode=False):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node.unsqueeze(-1)), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected.unsqueeze(-1)
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        assert (self.load >= -self.round_error_epsilon).all(), "load cannot be negative!"
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag.scatter_(dim=-1, index=gathering_index, value=float('-inf'))
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()

        demand_too_large = self.load[:, :, None] + self.round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance(lib_mode)  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done

    def _get_travel_distance(self,lib_mode):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        if not lib_mode:
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            # shape: (batch, pomo, problem+1, 2)
        else:
            assert self.original_depot_node_xy_lib.size(0) == 1, 'The original_node_xy_lib should be a single instance.'
            self.original_depot_node_xy_lib = self.original_depot_node_xy_lib.expand(self.batch_size, -1, -1)  # shape:(8,problem+1,2)
            all_xy = self.original_depot_node_xy_lib[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            # shape: (8, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def get_local_feature(self):
        # dist.shape: (batch, problem+1, problem+1)
        # current_node.shape: (batch, pomo)
        if self.current_node is None:
            return None

        current_node = self.current_node.unsqueeze(-1).expand(-1, -1, self.problem_size + 1)
        # shape: (batch, pomo, problem+1)
        cur_dist = self.dist.gather(dim=1,index=current_node)
        # shape: (batch, pomo, problem+1)

        return cur_dist


