import pickle

import torch
import numpy as np
from tqdm import tqdm


def get_random_problems_cvrp(batch_size, problem_size,capacity):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    demand = torch.randint(1, 10, size=(batch_size, problem_size))
    # shape: (batch, problem)

    node_demand = demand / float(capacity)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

def get_saved_data(filename, total_episodes,device, start=0, solution_name=None):

    data_type = filename.split('.')[-1]

    data_loader = {
        'pkl': use_saved_problems_cvrp_pkl,
        'txt': use_saved_problems_cvrp_txt,
    }

    if data_type not in data_loader.keys():
        assert False, f"Unsupported file type: {data_type}. Supported types are: {list(data_loader.keys())}"

    return data_loader[data_type](filename, total_episodes, device, start, solution_name)

def use_saved_problems_cvrp_pkl(filename, total_episodes,device, start=0, solution_name=None):
    with open(filename, 'rb') as f1:
        out_1 = pickle.load(f1)[start:start + total_episodes]
        out = np.array(out_1, dtype=object)
        raw_data_depot = torch.tensor(out[:, 0].tolist(), dtype=torch.float32).to(device)
        if raw_data_depot.dim() == 2:
            raw_data_depot = raw_data_depot[:, None, :] # shape: (batch, 1, 2)
        raw_data_nodes = torch.tensor(out[:, 1].tolist(), dtype=torch.float32).to(device)
        # shape: (batch, problem, 2)
        raw_data_demand = torch.tensor(out[:, 2].tolist(), dtype=torch.float32)
        # shape: (batch, problem)
        capacity = float(out[0, 3])
        raw_data_demand = (raw_data_demand / capacity).to(device)
    if solution_name is not None:
        with open(solution_name, 'rb') as f2:
            out_2 = pickle.load(f2)[start:total_episodes]
            out_2 = np.array(out_2, dtype=object)[:, 0].tolist()
            optimal_score_all = torch.tensor(out_2, dtype=torch.float32, device=device)
            optimal_score = optimal_score_all.mean().item()
    else:
        # if no optimal score, please manually give an average optimal value used for calculating the gap.
        optimal_score = 1.0

    dataset_dict = {
        'depot_xy': raw_data_depot,
        'node_xy': raw_data_nodes,
        'node_demand': raw_data_demand,
        'capacity': capacity,
    }

    return dataset_dict, optimal_score

def use_saved_problems_cvrp_txt(filename, total_episodes,device, start=0, solution_name=None):
    raw_data_nodes = []
    raw_data_depot = []
    raw_data_demand = []
    raw_cost = []
    capacity = 0

    for line in tqdm(open(filename, "r").readlines()[start:start + total_episodes], ascii=True):
        line = line.split(",")

        depot_index = int(line.index('depot'))
        customer_index = int(line.index('customer'))
        capacity_index = int(line.index('capacity'))
        demand_index = int(line.index('demand'))
        cost_index = int(line.index('cost'))

        depot = [[float(line[depot_index + 1]), float(line[depot_index + 2])]]
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in
                    range(customer_index + 1, capacity_index, 2)]
        raw_data_nodes.append(customer)
        raw_data_depot.append(depot)

        if capacity == 0:
            capacity = float(line[capacity_index + 1])

        demand = [int(line[idx]) for idx in range(demand_index + 1, cost_index)]
        raw_data_demand.append(demand)
        raw_cost.append(float(line[cost_index + 1]))

    raw_data_depot = torch.tensor(raw_data_depot, device=device)
    # shape: (batch, 1, 2)
    raw_data_nodes = torch.tensor(raw_data_nodes, device=device)
    # shape: (batch, problem, 2)
    raw_data_demand = torch.tensor(raw_data_demand, device=device) / capacity
    # shape: (batch, problem)
    optimal_score = np.mean(raw_cost)

    dataset_dict = {
        'depot_xy': raw_data_depot,
        'node_xy': raw_data_nodes,
        'node_demand': raw_data_demand,
        'capacity': capacity,
    }

    return dataset_dict, optimal_score