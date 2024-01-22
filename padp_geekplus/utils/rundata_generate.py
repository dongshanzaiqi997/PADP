import numpy as np
import torch
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from datetime import datetime

from Config import Config
from Model import ActorCritic
from StateModel import StateModel
from Train import Train
from Test import Test
from tqdm import trange
from tensorboardX import SummaryWriter
import pandas as  pd

def main():
    ac = ActorCritic(Config.num_inputs, Config.num_control, stochastic_policy=True)
    state_model = StateModel()
    test = Test()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    ac.load_net('2020-11-14T13-37-38')
    test.evaluate_nn(ac, state_model,bad_episode=not True)
    test.plot_nn()
    df = pd.DataFrame(test.test_x_list)
    df.to_csv('data/MF_0.9/run/' + TIMESTAMP+'.csv')


if __name__ == '__main__':
    main()




