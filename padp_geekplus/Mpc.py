from casadi import *

import numpy as np
from Config import Config
import torch


class Dynamics(object):
    def __init__(self, ):
        self.L = 4.  # length of the vehicle
        self.W = 1.7  # width of the vehicle
        self.laneWidth = 4
        self.exp_v = 7
        self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
                                   C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
                                   a=1.19,  # distance from CG to front axle [m]
                                   b=1.46,  # distance from CG to rear axle [m]
                                   mass=1520.,  # mass [kg]
                                   I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, states, actions, tau=1/Config.num_frequency):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi = states[0], states[1], states[2], states[3], states[4], states[5]
        steer, a_x = actions[0], actions[1]

        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * power(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (a ** 2 * C_f + b ** 2 * C_r) - I_z * v_x),
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      (phi + tau * r)]

        return next_state

    def tracking_aug(self, x):
        # left
        # if x[4]<-self.laneWidth:
        #     delta_position = x[3] - self.laneWidth / 2
        # elif x[4]>-self.laneWidth and x[3]>-self.laneWidth:
        #     delta_position = sqrt(power(x[:, 3] - (-self.laneWidth),2) + power(x[:, 4] - (-self.laneWidth)),2) - 1.5*self.laneWidth
        # else:
        #     delta_position = x[4] - self.laneWidth / 2
        delta_ = sqrt(power(x[3] - (-self.laneWidth),2) + power(x[4] - (-self.laneWidth),2)) - 1.5*self.laneWidth
        delta_ = if_else(x[4] < -self.laneWidth, x[3] - self.laneWidth / 2, delta_)
        delta_position = if_else(x[3] < -self.laneWidth, x[4] - self.laneWidth / 2, delta_)

        #straight
        #delta_position = x[3] - self.laneWidth / 2

        # left
        delta_ = np.pi/2 + atan2(x[4]+self.laneWidth, x[3]+self.laneWidth) - x[5]
        delta_ = if_else(x[4] < -self.laneWidth, np.pi/2 - x[5], delta_)
        delta_ = if_else(x[3] < -self.laneWidth, np.pi - x[5], delta_)
        delta_ = if_else(delta_ > np.pi, delta_ - np.pi * 2, delta_)
        delta_head = if_else(delta_ < -np.pi, delta_ + np.pi * 2, delta_)

        # if x[4]<-self.laneWidth:
        #     delta_head = np.pi/2 - x[5]
        # elif x[4]>-self.laneWidth and x[3]>-self.laneWidth:
        #     delta_head = np.pi/2 + torch.atan2(x[4]+self.laneWidth, x[3]+self.laneWidth) - x[5]
        # else:
        #     delta_head = np.pi - x[:,5]

        # straight
        # delta_head = np.pi/2 - x[5]
        delta_v = x[0] - self.exp_v

        x_aug = x + [delta_position] + [delta_head] + [delta_v]
        return x_aug

    def step(self, x, u):
        x_next = self.f_xu(x, u)
        x_aug = self.tracking_aug(x_next)
        return x_aug

    def g_x(self, x):
        g_list = []
        lws = (self.L - self.W) / 2
        ego_front_points = x[3] + lws * cos(x[5]), x[4] + lws * sin(x[5])
        ego_rear_points = x[3] - lws * cos(x[5]), x[4] - lws * sin(x[5])

        for veh_index in range(2):
            vehs = self.vehs
            veh_front_points = vehs[0] + lws, vehs[1]
            veh_rear_points = vehs[0] - lws, vehs[1]
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = sqrt(
                        power(ego_point[0] - veh_point[0], 2) + power(ego_point[1] - veh_point[1], 2))
                    veh2veh_dist = 3-veh2veh_dist
                    g_list.append(veh2veh_dist)

        lws = 0.85

        for ego_point in [ego_front_points]:
            g_list.append(if_else(ego_point[1] < -self.laneWidth, lws-ego_point[0], -1))
            g_list.append(if_else(ego_point[1] < -self.laneWidth, lws+ego_point[0]-self.laneWidth, -1))
            g_list.append(if_else((ego_point[0]<-self.laneWidth), lws-ego_point[1],-1))
            g_list.append(if_else((ego_point[0]<-self.laneWidth), lws+ego_point[1]-self.laneWidth,-1))
            # g_list.append(if_else(ego_point[1] > self.laneWidth, lws+-ego_point[0],-1))
            # g_list.append(if_else(ego_point[1] > self.laneWidth, lws+ego_point[0]-self.laneWidth,-1))
        for ego_point in [ego_rear_points]:
            g_list.append(if_else(ego_point[1] < -self.laneWidth, lws-ego_point[0], -1))
            g_list.append(if_else(ego_point[1] < -self.laneWidth, lws+ego_point[0]-self.laneWidth, -1))
            g_list.append(if_else((ego_point[0]<-self.laneWidth), lws-ego_point[1], -1))
            g_list.append(if_else((ego_point[0]<-self.laneWidth), lws+ego_point[1] - self.laneWidth, -1))
            # g_list.append(if_else(ego_point[1] > self.laneWidth, lws-ego_point[0], -1))
            # g_list.append(if_else(ego_point[1] > self.laneWidth, lws+ego_point[0] - self.laneWidth, -1))

        return g_list

    def vehs_pre(self):
        step_size = 1/Config.num_frequency
        x = self.vehs
        x[0] = x[0] + step_size * x[2]
        x[1] = x[1]

        x[3] = x[3] - step_size * x[5]
        x[4] = x[4]


class ModelPredictiveControl(object):
    def __init__(self,):
        self.horizon = Config.num_rollout_step
        self.base_frequency = 10.
        self.exp_v = 7
        self.DYNAMICS_DIM = 9  # ego_info + track_error_dim
        self.ACTION_DIM = 2
        self.dynamics = Dynamics()
        self._sol_dic = {'ipopt.print_level': 0,
                         'ipopt.sb': 'yes',
                         'print_time': 0}

    def mpc_solver(self, x_init):
        x_init = x_init.tolist()[0]
        self.x0 = np.zeros((339))
        x = SX.sym('x', self.DYNAMICS_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        # Create empty NLP
        w = []
        lbw = []  # lower bound for state and action constraints
        ubw = []  # upper bound for state and action constraints
        lbg = []  # lower bound for distance constraint
        ubg = []  # upper bound for distance constraint
        G = []  # dynamic constraints
        J = 0  # accumulated cost

        # Initial conditions
        Xk = MX.sym('X0', self.DYNAMICS_DIM)
        w += [Xk]
        lbw += x_init[:6] + x_init[12:]
        ubw += x_init[:6] + x_init[12:]
        self.dynamics.vehs = x_init[6:12]

        for k in range(1, self.horizon + 1):
            f = vertcat(*self.dynamics.step(x, u))
            F = Function("F", [x, u], [f])
            g = vertcat(*self.dynamics.g_x(x))
            G_f = Function('Gf', [x], [g])

            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += [-np.pi/6, -3.]
            ubw += [np.pi/6, 3.]

            Fk = F(Xk, Uk)
            Gk = G_f(Xk)
            self.dynamics.vehs_pre()
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.DYNAMICS_DIM)

            # Dynamic Constraints
            G += [Fk - Xk]  # ego vehicle dynamic constraints
            lbg += [0.0] * self.DYNAMICS_DIM
            ubg += [0.0] * self.DYNAMICS_DIM
            G += [Gk]  # surrounding vehicle constraints
            lbg += [-inf] * (2 * 4 + 2*4)
            ubg += [0.] * (2 * 4 ) + [0]*2*4
            w += [Xk]
            lbw += [0.] + [-inf] * (self.DYNAMICS_DIM - 1)  # speed constraints
            ubw += [15.] + [inf] * (self.DYNAMICS_DIM - 1)

            # Cost function
            F_cost = Function('F_cost', [x, u], [0.05 * power(x[8], 2)
                                                 + 0.8 * power(x[6], 2)
                                                 + 30 * power(x[7], 2)
                                                 + 0.2 * power(x[2], 2)
                                                 + 20 * power(u[0], 2)
                                                 + 0.05 * power(u[1], 2)
                                                 ])
            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # load constraints and solve NLP
        r = S(lbx=vertcat(*lbw), ubx=vertcat(*ubw),x0=self.x0, lbg=vertcat(*lbg), ubg=vertcat(*ubg))
        state_all = np.array(r['x'])
        self.x0 = np.concatenate((state_all[11:],state_all[-11:]))
        g_all = np.array(r['g'])
        state = np.zeros([self.horizon, self.DYNAMICS_DIM])
        control = np.zeros([self.horizon, self.ACTION_DIM])
        nt = self.DYNAMICS_DIM + self.ACTION_DIM  # total variable per step
        cost = np.array(r['f']).squeeze(0)

        # save trajectories
        for i in range(self.horizon):
            state[i] = state_all[nt * i: nt * (i + 1) - self.ACTION_DIM].reshape(-1)
            control[i] = state_all[nt * (i + 1) - self.ACTION_DIM: nt * (i + 1)].reshape(-1)
        return state, control, state_all, g_all, cost

if __name__ == "__main__":
        from StateModel import CrossRoadsEnv
        state_model = CrossRoadsEnv()
        mpc = ModelPredictiveControl()
        x = state_model.reset(1)
        for i in range(90):
            # print('vx, vy, r')
            # print(x[0, :3])
            # print('x, y, phi')
            # print(x[0, 3:6])
            print('dx, dph, dv')
            print(x[0, -3:])

            #u = ac.step(x)
            _,u,_,_,_ = mpc.mpc_solver(x)
            x, r, c, die, done = state_model.step(x, torch.Tensor(u[0:1]))
            state_model.render(x)
            print('action')
            print(u[0])