import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import Config
import datetime
from torch.optim import Adam
import numpy as np
from torch.distributions.normal import Normal

seeds = [2,29,31,35, 27]
#torch.manual_seed(5) #37
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class GaussianActor(nn.Module, Config):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -2. * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], nn.Tanh)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        if act is not None:
            logp_a = pi.log_prob(act)
            return logp_a
        a = torch.clamp(pi.sample(), self.min_acc, self.max_acc)
        return a

    def _distribution(self, obs):
        mu = self.mu_net(obs) * self.acc_coe + self.acc_bias
        std = torch.exp(self.log_std)
        return Normal(mu, std)


class Actor(nn.Module, Config):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.pi_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh)

    def forward(self, obs):
        obs_relative = torch.cat((obs[:, :3]-obs[:, 5:8], obs[:, 3:5], obs[:, 8:]), 1) / \
                       torch.Tensor([[7, 7, np.pi, 0.5, np.pi / 2, 0.5, np.pi / 2, 7, np.pi, 1]])
        # obs = obs / torch.Tensor([[7, 7, np.pi, 0.5, np.pi/2, 7, 7, np.pi, 0.5, np.pi/2, 7, np.pi, 1]])
        a = self.pi_net(obs_relative)*torch.Tensor([0.2, np.pi/2]) + torch.Tensor([0.2,0])
        return a

class QValue(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.fcs = nn.Linear(obs_dim, 64)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(act_dim, 64)
        self.fca.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(64, 32)
        self.out.weight.data.normal_(0, 0.1)
        self.out2 = nn.Linear(32, 1)
        self.out2.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x + y)
        actions_value = self.out2(F.relu(self.out(net)))
        return actions_value


class SValue(nn.Module):

    def __init__(self, obs_dim):
        super().__init__()
        self.fcs = nn.Linear(obs_dim, 256)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization

        self.out = nn.Linear(256, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, obs):
        obs = torch.cat((obs[:, :3]-obs[:, 5:8], obs[:, 3:5], obs[:, 8:]), 1) / \
                       torch.Tensor([[7, 7, np.pi, 0.5, np.pi / 2, 0.5, np.pi / 2, 7, np.pi, 1]])
        # obs = obs / torch.Tensor([[7, 7, np.pi, 0.5, np.pi/2, 7, 7, np.pi, 0.5, np.pi/2, 7, np.pi, 1]])

        x = self.fcs(obs)
        x = F.relu(x)
        x = self.out(x)
        safe_prob = F.tanh(x) * 1.0
        return safe_prob


class ActorCritic(nn.Module, Config):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64, 64), activation=nn.ReLU, stochastic_policy=False):
        super().__init__()
        if stochastic_policy:
            self.pi = GaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        else:
            self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation)
        self.q = QValue(obs_dim, act_dim, hidden_sizes, activation)
        self.qs = SValue(obs_dim)
        self.pi_optimizer = Adam(self.pi.parameters(), lr=self.learning_rate_policy)
        self.qs_optimizer = Adam(self.qs.parameters(), lr=self.learning_rate_qs)

    def step(self, obs):
        with torch.no_grad():
            a = self.pi(obs)
        return a


    def save_net(self, name):
        torch.save(self.pi.state_dict(), 'logs/train/' + name + '/'+ name +'_policy-ccrl' + '.pkl')
        # torch.save(self.q.state_dict(),
        #            name + '_value-ccrl' + '.pkl')
        torch.save(self.qs.state_dict(), 'logs/train/' + name + '/'+ name +'_qs-ccrl' + '.pkl')

    def load_net(self, index, unsafe_init=False):
        #self.v.load_state_dict(torch.load('2020-09-28, 22.56.29_value-moeldriven_40000.pkl'))
        if unsafe_init:
            self.pi.load_state_dict(torch.load('network/'+index+'_policy-ccrl.pkl'))
        else:
            self.pi.load_state_dict(torch.load('logs/train/'+index+'/'+index+'_policy-ccrl.pkl'))
            self.qs.load_state_dict(torch.load('logs/train/'+index+'/'+index+'_qs-ccrl.pkl'))


if __name__ == "__main__":
    ac = ActorCritic(Config.num_inputs-3, Config.num_control)
    ac.load_net('2021-05-29T00-49-15')
    obs = torch.Tensor([[ 0.5297,  1.1331, -0.0566,  0.0000,  0.0000,  3.5000, -1.0000, -1.5708,
          0.0000,  0.0000, -0.2669, -0.0566, -0.3000]])

    print(ac.step(obs))











