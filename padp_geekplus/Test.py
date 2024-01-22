from tensorboardX import SummaryWriter
import torch
import numpy as np
from Config import Config
import matplotlib.pyplot as plt
import datetime
# from Mpc import ModelPredictiveControl
from mpl_toolkits.mplot3d import Axes3D


class Test(Config):
    def __init__(self, writer):
        super(Test, self).__init__()
        self.writer = writer
        self.evaluate_index = 0
        #self.mpc = ModelPredictiveControl()


    def render_episode(self, ac, state_model):
        x = state_model.reset(256)
        for i in range(100):
            # print('vx, vy, r')
            # print(x[0, :3])
            # print('x, y, phi')
            # print(x[0, 3:6])
            # print('dx, dph, dv')
            # print(x[0, -3:])

            u = ac.step(x)
            #_,u,_,_,_ = self.mpc.mpc_solver(x,x)
            x, r, c, die, done, _ = state_model.step(x, u)
            state_model.render(x)
            print('action')
            print(u[0])
            # print('x')
            # print(x[0])
            # print('r')
            # print(r[0])
            print(torch.cat((x[0, 5:8]-x[0, :3], x[0, 3:5], x[0, 8:]), 0))
            print((((x[0, 5:7] - x[0, :2])**2).sum())**0.5)
        #state_model.close()


    def evaluate_train(self, ac, state_model):
        x_init = state_model.reset(Config.num_evaluate_agent)
        x = torch.from_numpy(x_init).float()
        live = torch.ones([x.size()[0], 1])
        cum_reward = torch.zeros([x.size()[0], 1])
        for i in range(Config.num_test_max_step):
            u = ac.predict(x, )
            x, r, rk, dead = state_model.step_env(x, u)
            cum_reward += r * live
            live = live * torch.Tensor((1-dead))
        self.writer.add_scalar('train/env_safe_prob', live.mean().item(), self.evaluate_index * self.evaluate_frequency)
        self.writer.add_scalar('train/env_r_sum', cum_reward.mean().item(), self.evaluate_index * self.evaluate_frequency)
        print('env: live=%.4f  return=%.1f' % (live.mean().item(), cum_reward.mean().item()))
        self.evaluate_index += 1

    def evaluate_nn(self, ac, state_model, bad_episode=False):
        x_init = state_model.reset(num_agent=Config.num_test_agent)
        x = torch.from_numpy(x_init).float()
        live = torch.ones([x.size()[0], 1])
        self.test_x_list = np.zeros([Config.num_test_max_step, Config.num_test_agent, Config.num_inputs])
        self.test_r_list = np.zeros([Config.num_test_max_step, Config.num_test_agent, 1])
        self.test_u_list = np.zeros([Config.num_test_max_step, Config.num_test_agent, 1])
        cum_reward = 0
        ave_gap = 0
        for i in range(Config.num_test_max_step):
            u = ac.predict(x, )
            x, r, rk, dead = state_model.step_env(x, u)
            live = live * torch.Tensor((1 - dead))
            self.test_x_list[i] = x.detach().numpy()
            self.test_r_list[i] = r.detach().numpy()
            self.test_u_list[i] = u.detach().numpy()
            cum_reward += r
            ave_gap += np.clip(x[:,2].detach().numpy(), 0, 99).mean()
        print(live.mean().numpy())
        print(ave_gap/Config.num_test_max_step)
        print(cum_reward.mean() / Config.num_test_max_step)
        if bad_episode:
            bad_index = torch.where(live<=0)[0]
            self.test_x_list = self.test_x_list[:,bad_index]
            self.test_r_list = self.test_r_list[:,bad_index]
            self.test_u_list = self.test_u_list[:,bad_index]
        # else:
        #     self.test_x_list = self.test_x_list[:, 0]
        #     self.test_r_list = self.test_r_list[:, 0]
        #     self.test_u_list = self.test_u_list[:, 0]


    def evaluate_nn_dis(self, ac, state_model, d_s, policy, train=None):
        estimate_num = 200
        ds = np.loadtxt(str(d_s)+'d_s.csv')
        #ds = state_model.random_input(num_agent=256*4)
        if train != None:
            ds = train.x
            thre = Config.safe_prob
        else:
            thre = policy
            if policy==0.9:
                ac.load_net('2021-04-06T14-45-570.9')
            else:
                ac.load_net('2021-04-06T14-56-140.999')
        x_init = np.tile(ds, (estimate_num, 1))
        #x_init = np.tile(state_model.random_input(256), (estimate_num, 1))
        x = torch.from_numpy(x_init).float()
        live = torch.ones([x.size()[0], 1])
        cum_reward = 0
        for i in range(Config.num_test_max_step):
            u = ac.predict(x, )
            x, r, rk, dead = state_model.step_env(x, u)
            live = live * torch.Tensor((1 - dead))
            cum_reward += r
        live_es = [];
        cum_es = []
        index_st = np.linspace(0, x_init.shape[0] - x_init.shape[0] / estimate_num, estimate_num, dtype=int)
        for i in range(int(x_init.shape[0] / estimate_num)):
            index = index_st + i
            live_es.append(live[index].mean())
            cum_es.append(cum_reward[index].mean())

            # test the accuracy of MC sampling

            # live_mean = live[index]
            # live_list = np.zeros((live_mean.shape[0]))
            # for j in range(live_mean.shape[0]):
            #     live_list[j] = live_mean[:j+1].mean()
            # if live_list[-1]<1:
            #     plt.figure()
            #     plt.plot(np.arange(1, live_mean.shape[0]+1), live_list)
            #     plt.ylim(0,1)
            #     plt.show()

        # plt.figure()
        # live_list = np.array(live_es)
        # for j in range(live_list.shape[0]):
        #     live_list[j] = np.array(live_es[:j+1]).mean()
        # plt.plot(np.arange(1, live_list.shape[0]+1), live_list)
        # plt.ylim(0,1)
        #


        live_es = np.array(live_es)
        cum_es = np.array(cum_es)
        print(str(policy)+'policy on '+str(d_s)+' dis')

        print(live_es.mean())
        print(cum_es.mean())
        print('-----------------------')
        plt.figure()
        plt.hist(live_es, bins=20)
        plt.xlabel('live_prob')
        plt.xlim(0, 1)

        # plt.figure()
        # plt.hist(cum_es, bins=20)
        # plt.xlabel('reward')
        ax = Axes3D(plt.figure())
        ax.scatter(ds[live_es > thre, 0], ds[live_es > thre, 1], ds[live_es > thre, 2], color='green')
        ax.scatter(ds[live_es <= thre, 0], ds[live_es <= thre, 1], ds[live_es <= thre, 2], color='red')
        ax.set_zlabel('gap')
        ax.set_ylabel('vf')
        ax.set_xlabel('ve')
        ax.set_zlim(1.8, 18)
        plt.title(str(policy)+'policy on '+str(d_s)+' dis')
        plt.show()

    def plot_nn(self):
        test_step = np.linspace(1, Config.num_test_max_step, Config.num_test_max_step)
        plt.figure()
        for i in range(12):
            plt.subplot(3, 4, i+1)
            plt.plot(test_step, self.test_x_list[:, i, 0], linewidth=2.0, label="ego vehicle speed")
            plt.plot(test_step, self.test_x_list[:, i, 1], linewidth=2.0, label="target vehicle")
            plt.plot(test_step, self.test_x_list[:, i, 2], linewidth=2.0, label="gap")
            plt.plot(test_step, self.test_u_list[:, i], linewidth=2.0, label="action")
            # plt.plot(test_step, self.test_r_list, linewidth=2.0, label="test_r")
            # plt.plot(test_step, self.test_u_list, linewidth=2.0, label="test_u")
            plt.ylim(-2,10)
        plt.legend(loc='lower right')
        plt.xlabel('step')
        plt.title('Test')
        plt.show()


# def test():
#
#     a = torch.tensor([1,-0.1,0.1,2,-0.2, 0.2])
#     a[torch.abs(a)<0.15] = 0
#     print(a)
#
#     pass
#
#
#
#
# if __name__ == "__main__":
#     test()