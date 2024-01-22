import torch
import numpy as np
from Config import Config
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Robot():
    def __init__(self, path=None):
        self.robot_params = dict(  v_max=0.4, 
                                   w_max=np.pi / 2,  
                                   v_delta_max=1.8,  # per second
                                   w_delta_max=0.8,
                                   v_desired=0.3,
                                   radius=0.74 / 2
                                   )
        self.path = path

    def f_xu(self, states, actions, T, type):
        v_delta_max = self.robot_params['v_delta_max']
        v_max = self.robot_params['v_max']
        w_max = self.robot_params['w_max']
        w_delta_max = self.robot_params['w_delta_max']
        dis = {'ego':[0.08,0.05],'obs':[0.07, 0.03],'none':[0, 0],'explore':[0.3, 0.3]}
        stds = dis[type]
        
        x, y, theta, v, w = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
        v_cmd, w_cmd = actions[:, 0], actions[:, 1]

        delta_v = torch.clamp(v_cmd-v, -v_delta_max*T, v_delta_max*T)
        delta_w = torch.clamp(w_cmd - w, -w_delta_max * T, w_delta_max * T)
        v_cmd = torch.clamp(v+delta_v, -v_max, v_max) + torch.Tensor(np.random.normal(0, stds[0], [states.shape[0]]))*0.5
        w_cmd = torch.clamp(w+delta_w, -w_max, w_max) + torch.Tensor(np.random.normal(0, stds[1], [states.shape[0]]))*0.5
        next_theta = theta + T*w_cmd
        next_theta = torch.where(next_theta > np.pi, next_theta - np.pi * 2, next_theta)
        next_theta = torch.where(next_theta < -np.pi, next_theta + np.pi * 2, next_theta)
        next_state = [x + T*torch.cos(theta)*v_cmd,
                     y + T*torch.sin(theta)*v_cmd,
                     next_theta,
                     v_cmd,
                     w_cmd]

        return torch.stack(next_state, 1)

    def tracking_error(self, x):
        delta_position = x[:, 1]
        delta_head = x[:, 2]
        # print('delta_head', delta_head.detach())
        delta_head = torch.where(delta_head > np.pi, delta_head - np.pi * 2, delta_head)
        delta_head = torch.where(delta_head < -np.pi, delta_head + np.pi * 2, delta_head)

        delta_v = x[:, 3] - self.robot_params['v_desired']
        # print('delta_position', torch.norm(delta_position.detach())**2/1024)
        # print('delta_head', delta_head.detach())
        # print('delta_v', torch.norm(delta_v.detach())**2/1024)
        tracking = torch.cat((delta_position.reshape(-1, 1), delta_head.reshape(-1, 1), delta_v.reshape(-1, 1)), 1)
        return tracking


class Environment(Config):
    def __init__(self, ):

        self.robot = Robot()
        self.obstacle = Robot()

        fig, axs = plt.subplots(3, 3, figsize=(6, 6))
        circles = []
        arrows = []
        r_rob = self.robot.robot_params['radius']
        r_obs = self.obstacle.robot_params['radius']
        for i in range(3):
            for j in range(3):
                ax = axs[i, j]
                ax.set_aspect(1)
                ax.set_ylim(-3, 3)
                #ax.cla()
                ax.plot([0, 6],[0, 0], "k")
                circles.append(plt.Circle([0, 0], r_rob, color='red', fill=False))
                ax.add_artist(circles[-1])
                circles.append(plt.Circle([0, 0], r_obs, color='blue', fill=False))
                ax.add_artist(circles[-1])

                arrows.append(ax.plot([], [], "red")[0])
                arrows.append(ax.plot([], [], "blue")[0])
        self.circles = circles
        self.arrows = arrows
        plt.ion()

    def reset(self, n_agent=Config.num_agent):
        def uniform(low, high):
            return torch.rand([n_agent])*(high-low) + low
        x = torch.zeros([n_agent, 10])
        x[:, 0] = uniform(0, 2.7)
        x[:, 1] = uniform(-1, 1)
        x[:, 2] = uniform(-0.6, 0.6)
        x[:, 3] = uniform(0, 0.3)
        x[:, 4] = x[:, 4]

        x[:, 5] = uniform(3.5, 6)
        x[:, 6] = uniform(-3, 3)
        x[:, 7] = torch.where(x[:, 6] > 0, x[:, 7]-np.pi/2, x[:, 7]+np.pi/2) + uniform(-0.8, 0.8)
        x[:, 8] = uniform(0., 0.5)
        x[:, 9] = x[:, 9]

        tracking = self.robot.tracking_error(x[:, :5])
        return torch.cat((x, tracking), 1)

    def step(self, x, u, T=0.2):
        x_ego = self.robot.f_xu(x[:, :5], u, T, 'ego')
        x_obs = self.obstacle.f_xu(x[:, 5:10], x[:, 8:10], T, 'none')
        tracking = self.robot.tracking_error(x_ego)
        x = torch.cat((x_ego, x_obs, tracking), 1)

        r_tracking = -1.4 * torch.square(x[:, -3]) - 1 * x[:, -2] ** 2 - 16*x[:, -1] ** 2
        # print('r_teacking', r_tracking.mean().detach())
        r_action = - 0.2 * u[:, 0] ** 2 - 0.5 * u[:, 1] ** 2
        # print('action', u[:, 0].mean().detach(), u[:, 1].mean().detach())
        reward = (r_tracking + r_action).reshape(-1, 1)

        # computing cost and dead
        safe_dis = self.robot.robot_params['radius'] + self.obstacle.robot_params['radius'] + 0.15  #0.35
        veh2vehdist = safe_dis - (torch.sqrt(torch.square(x[:, 5] - x[:, 0]) + torch.square(x[:, 6] - x[:, 1]))).reshape(-1, 1)

        def Phi(y):
            m1 = 1
            m2 = m1 / (1 + m1) * 0.9
            #m2 = 3/2
            tau = 0.07
            sig = (1 + tau * m1) / (1 + m2 * tau * torch.exp(torch.clamp(y / tau, min=-10, max=5)))

            # c = torch.relu(-y)
            return sig

        it = 99999
        r_penalty = -min(200, 6+max(0, it-400)*100/1000)*torch.square(torch.relu(veh2vehdist))
        reward = reward


        assert (reward > -1e8).all()
        cost = torch.zeros([x.shape[0], 1])
        dead = torch.zeros([x.shape[0], 1]).detach()
        for i in range(1):
            cost[:, i] = Phi(veh2vehdist[:, i])
            dead[:, i] = veh2vehdist[:, i] > 0

        done = (x[:, 0] < 0) + (x[:, 0] > 7) + (x[:, 1] > 3) + (x[:, 1] < -1)
        done = done.reshape(-1, 1)

        return x, reward, cost, dead, done, [r_tracking.reshape(-1, 1), r_penalty, veh2vehdist]

    def render(self, x):
        r_rob = self.robot.robot_params['radius']
        r_obs = self.obstacle.robot_params['radius']
        def arrow_pos(state):
            x, y, theta = state[0], state[1], state[2]
            return [x, x+torch.cos(theta)*r_rob], [y, y+torch.sin(theta)*r_rob]


        for i in range(3):
            for j in range(3):
                self.circles[i * 6 + j * 2].center = (x[i * 2 + j, 0], x[i * 2 + j, 1])
                self.circles[i * 6 + j * 2 + 1].center = (x[i * 2 + j, 5], x[i * 2 + j, 6])
                self.arrows[i * 6 + j * 2].set_data(arrow_pos(x[i * 2 + j, :5]))
                self.arrows[i * 6 + j * 2+1].set_data(arrow_pos(x[i * 2 + j, 5:10]))
        plt.pause(0.01)

    def close(self):
        plt.close('all')



def tes_env():
    env = Environment()
    x = env.reset(9)
    for i in range(1000):
        u = torch.zeros([9, 2])
        u[:, 0] = 0.4
        u[:, 1] = -x[:, 10]
        x, r, c, die, done = env.step(x, u)

        #print(np.array(x)[0,3:6])
        print(r[0])
        print(c[0])
        print(die[0])
        print(done[0])
        # print('left')
        # print(np.array(x)[0,6:9])
        # print('up')
        # print(np.array(x)[0,9:12])
        env.render(x)


if __name__ == "__main__":
    tes_env()