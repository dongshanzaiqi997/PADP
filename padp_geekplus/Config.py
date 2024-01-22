import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from torch.distributions.normal import Normal


class Config:
    # general
    num_inputs = 13
    num_control = 2

    # algorithm
    learning_rate_q = 2e-3
    learning_rate_qs = 5e-2

    learning_rate_policy = 36e-5
    # learning_rate_policy = 1.2e-5
    learning_rate_decay = 1

    gamma = 1

    # train
    num_iteration = 30000
    num_agent = 256*4
    num_rollout_agent = 4*256
    num_max_step = (40 + np.floor(np.random.uniform(0, 140, [num_agent, 1])))
    # num_rollout_step = int(0.5 * num_frequency)
    num_rollout_step = 25
    num_buffer = num_agent * 400
    num_rollout_branch = 4
    save_frequency = 30000

    # evaluate
    evaluate_frequency = 1000
    evaluate_times = num_iteration // evaluate_frequency + 1
    num_evaluate_agent = 2560*2
    num_evaluate_max_step = 100

    # test
    num_test_agent = 2560*4
    num_test_max_step = 40

    # state model parameters
    num_frequency = 2.5
    dt = 1 / num_frequency



    # model disturbance
    mu = 0.0
    var = 0.0
    max_d = 7
    min_d = -7

    n_constraint = 1

    cons_horizon = 80
    safe_prob = 0.99

    #primal-dual method
    lam_init = 2
    yita_lam = 7e-1


def test():
    pass

if __name__ == "__main__":
    test()
