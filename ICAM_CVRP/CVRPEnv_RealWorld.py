from dataclasses import dataclass

import math
import torch

from CVRProblemDef import augment_xy_data_by_8_fold
import numpy as np

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

        self.original_depot_node_xy = None # for lib data

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

    def input_saved_data(self,filename, total_episodes, problem_size, device, start=0, solution_name=None):
        data = torch.load(filename, map_location=device)

        loc_list = []
        dist_list = []
        demand_list = []
        capacity_list = []

        for k, v in data.items():
            loc_list.append(v['loc'])
            dist_list.append(v['dist'])
            demand_list.append(v['demand'])
            capacity_list.append(v['capacity'])

        loc_tensor = torch.stack(loc_list, dim=0)
        # shape: (total_episodes, problem_size+1, 2)
        assert loc_tensor.shape==(total_episodes, problem_size+1, 2)

        raw_data_depot = loc_tensor[:, [0], :].to(device)
        # shape: (total_episodes, 1, 2)
        raw_data_nodes = loc_tensor[:, 1:, :].to(device)
        # shape: (total_episodes, problem_size, 2)
        
        dist_tensor = torch.stack(dist_list, dim=0).to(device)
        # shape: (total_episodes, problem_size+1, problem_size+1)
        assert dist_tensor.shape==(total_episodes, problem_size+1, problem_size+1)
        
        raw_data_demand = torch.stack(demand_list, dim=0)
        # shape: (total_episodes, problem_size)
        capacity_tensor = torch.tensor(capacity_list, device=device)
        raw_data_demand = raw_data_demand / capacity_tensor.unsqueeze(1)
        capacity = capacity_tensor[0].item()
        solutions_costs = torch.load(solution_name, map_location=device) if solution_name is not None else None
        if solutions_costs is not None:
            solutions = solutions_costs['solution']
            # shape: (total_episodes, seq_len)
            costs = solutions_costs['cost'].to(device)
            # shape: (total_episodes,)
            
            # double check, calculate the costs based on the distance matrix
            node_from = solutions[:]
            seq_len = node_from.size(-1)
            # shape: (batch, node)
            node_to = solutions.roll(dims=1, shifts=-1)
            # shape: (batch, node)
            BATCH_IDX = torch.arange(total_episodes).to(device)
            batch_index = BATCH_IDX[:, None].expand(total_episodes, seq_len)
            # shape: (batch, node)

            selected_cost = dist_tensor[batch_index, node_from, node_to]
            #shape: (batch, node)
            
            total_distance = selected_cost.sum(1)
            #shape: (batch,)
            assert torch.allclose(total_distance, costs, atol=1e-4)
            
            optimal_score = costs.mean().item()
        
        else:
            # if no optimal score, please manually give an average optimal value used for calculating the gap.
            optimal_score = 1.0
        
        self.FLAG__use_saved_problems = True
        self.saved_depot_xy = raw_data_depot
        self.saved_node_xy = raw_data_nodes
        self.saved_node_demand = raw_data_demand
        self.saved_dist = dist_tensor
        self.capacity = capacity
        self.saved_index = 0
        self.device = device
        self.optimal_score = optimal_score

    def load_problems_cvrp_real_world(self, batch_size, problem_size,pomo_size=None,aug_factor=1, device=None):
        self.batch_size = batch_size
        self.problem_size = problem_size
        if pomo_size is None:
            self.pomo_size = problem_size
        else:
            self.pomo_size = pomo_size
        if device is not None:
            self.device = device

        
        depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
        node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
        node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
        self.saved_index += batch_size
        
        # we use the common coordinates normalization trick used in Att-GCN-MCTS and INViT papers.
        ##################################
        self.original_depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        xy_max = torch.max(self.original_depot_node_xy, dim=1, keepdim=True).values
        xy_min = torch.min(self.original_depot_node_xy, dim=1, keepdim=True).values
        # shape: (batch, 1, 2)
        ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
        ratio[ratio == 0] = 1
        # shape: (batch, 1, 1)
        problems = (self.original_depot_node_xy - xy_min) / ratio.expand(-1, 1, 2)
        # shape: (batch, dimension,2)
        
        depot_xy = problems[:, 0:1, :]
        node_xy = problems[:, 1:, :]
                
                
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                self.original_depot_node_xy = self.original_depot_node_xy.repeat(8, 1, 1)
                self.saved_dist = self.saved_dist.repeat(8, 1, 1)
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
        self.dist = self.distance_normalization()
        # shape: (batch, problem+1, problem+1)
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
            reward = - self._get_total_distance_by_dist()  # note the minus sign!
        else:
            reward = None
        return self.step_state, reward, done
    
    #calculate through dist
    def _get_total_distance_by_dist(self):
        node_from = self.selected_node_list
        seq_len = node_from.size(-1)
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        batch_index = BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, seq_len)
        # shape: (batch, pomo, node)

        selected_cost = self.saved_dist[batch_index, node_from, node_to]
        #shape: (batch, pomo, node)
        
        total_distance = selected_cost.sum(2)
        #shape: (batch, pomo)

        return total_distance
    
    def distance_normalization(self):
        # distance_matrix.shape: (batch, n, m)
        batch_size = self.saved_dist.size(0)
        dist_max = self.saved_dist.amax(dim=(1, 2), keepdim=True)
        assert dist_max.shape == (batch_size, 1, 1)
        dist_normed = self.saved_dist / (dist_max + 1e-8)  # normalize edge features per node
        
        return dist_normed * np.sqrt(2)

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
    
    


