import torch


def get_random_problems_atsp(batch_size, node_cnt, problem_gen_params):

    ################################
    # "tmat" type
    # Following MatNet: https://github.com/yd-kwon/MatNet/blob/main/ATSP/ATSProblemDef.py
    ################################

    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler

    return scaled_problems
    # shape: (batch, node, node)

def get_saved_data(filename, total_episodes,device, start=0, solution_name=None):

    data_type = filename.split('.')[-1]

    data_loader = {
        'pt': use_saved_problems_atsp_pt,
    }

    if data_type not in data_loader.keys():
        assert False, f"Unsupported file type: {data_type}. Supported types are: {list(data_loader.keys())}"

    return data_loader[data_type](filename, total_episodes, device, start, solution_name)

def use_saved_problems_atsp_pt(filename, total_episodes,device, start=0, solution_name=None):
    data = torch.load(filename, map_location=device)
    problems = torch.from_numpy(data['distance_matrix'])[start:start + total_episodes].to(device)
    optimal = data['optimal']
    return problems, optimal
