import torch
import numpy as np
from typing import Callable, Dict, List
from Config import Config
from utils.torch_utils import flat_grad, get_device, get_flat_params, normalize, set_params
from utils.distribution_utils import CategoricalDistribution
from tqdm import trange
from torch import nn
from copy import deepcopy
import math

EPSILON = 1e-8
rtol = 1e-5
atol = 1e-8
damping_factor = 0.01
max_cg = 10
delta = pow(0.001, 2)

class Train(Config):
    def __init__(self, writer):
        self.init_index = np.ones(self.num_agent, dtype=int)
        self.x = torch.zeros([self.num_agent, self.num_inputs])
        self.step = np.zeros([self.num_agent, 1])
        self.x_buffer = torch.zeros([self.num_buffer, self.num_inputs])
        self.buffer_index = 0
        self.buffer_full = False
        self.iteration_index = 0

        self.death_rate_pre = np.array([0.]*Config.n_constraint)
        self.delta_i = np.array([0.]*Config.n_constraint)
        self.writer = writer

        self.reward_min = 0.0
        self.reward_max = -999.0
        self.pf = 1
        self.pf_min = 1
        self.pf_max = 10
        self.ten_step_rew = []



    # def est_safe_prob(self, ac, state_model):
    #     with torch.no_grad():
    #         estimate_num = 4
    #         x = self.x_buffer[np.random.randint(0, self.num_buffer if self.buffer_full else self.buffer_index,
    #                                             size=[Config.num_rollout_agent])]
    #         x = x.repeat([estimate_num, 1])
    #         x_roll = x.clone()
    #         live = torch.ones([x.size()[0], 1])
    #         for i in range(self.num_rollout_step):
    #             u = ac.pi(x_roll)
    #             x_roll, r, c, dead, _, [r_tracking, r_penalty, hx] = state_model.step(x_roll, u)
    #             live = live * (1 - dead)
    #
    #         live_es = [];
    #         index_st = np.linspace(0, x.shape[0] - x.shape[0] / estimate_num, estimate_num, dtype=int)
    #         for i in range(int(x.shape[0] / estimate_num)):
    #             index = index_st + i
    #             live_es.append(float(live[index].mean()))
    #         live_es = torch.Tensor(live_es).reshape(-1, 1)
    #
    #     loss = ((live_es - ac.qs(x[:Config.num_rollout_agent])) ** 2).mean()
    #     # print(live_esti[:6])
    #     ac.qs_optimizer.zero_grad()
    #     loss.backward()
    #     ac.qs_optimizer.step()
    #     self.writer.add_scalar('train/safe_pro_loss', loss, self.iteration_index)
    #
    # def MC_safe_prob(self, ac, state_model):
    #     with torch.no_grad():
    #         estimate_num = 200
    #         x = self.x_buffer[np.random.randint(0, self.num_buffer if self.buffer_full else self.buffer_index,
    #                                             size=[32])]
    #         x = x.repeat([estimate_num, 1])
    #         x_roll = x.clone()
    #         live = torch.ones([x.size()[0], 1])
    #         for i in range(self.num_rollout_step):
    #             u = ac.pi(x_roll)
    #             x_roll, r, c, dead, _, [r_tracking, r_penalty, hx] = state_model.step(x_roll, u)
    #             live = live * (1 - dead)
    #
    #         live_es = [];
    #         index_st = np.linspace(0, x.shape[0] - x.shape[0] / estimate_num, estimate_num, dtype=int)
    #         for i in range(int(x.shape[0] / estimate_num)):
    #             index = index_st + i
    #             live_es.append(float(live[index].mean()))
    #         live_es = torch.Tensor(live_es).reshape(-1,1)
    #         live_compare = torch.cat((live_es, ac.qs(x[:32])),1)
    #         print(live_compare)
    #         #print(ac.qs(x[:estimate_num]))




    def PEV(self, ac, state_model):

        x = self.x_buffer[np.random.randint(0, self.num_buffer if self.buffer_full else self.buffer_index, size=[Config.num_rollout_agent])]
        x = x.repeat([1, 1])
        x_roll = x.clone()
        self.obs = x.clone()

        r_sum_N = torch.zeros([x.size()[0], 1])
        r_tracking_sum_N = torch.zeros([x.size()[0], 1])
        r_penalty_sum_N = torch.zeros([x.size()[0], 1])
        h_max_N = torch.zeros([x.size()[0], 1])
        c_sum_N = torch.zeros([x.size()[0], Config.n_constraint])
        c_multi = torch.ones([x.size()[0], Config.n_constraint])
        # dis_sum = torch.zeros([x.size()[0], Config.n_constraint])

        live = torch.ones([x.size()[0], 1])
        dead_N = torch.zeros([x.size()[0], self.num_rollout_step, Config.n_constraint])

        for i in range(self.num_rollout_step):
            u = ac.pi(x_roll)
            x_roll, r, c, dead, _, [r_tracking, r_penalty, hx] = state_model.step(x_roll, u)



            dead_N[:, i] = dead

            r_sum_N = r_sum_N + r * self.gamma ** i
            r_tracking_sum_N = r_tracking_sum_N + r_tracking * self.gamma ** i
            r_penalty_sum_N = r_penalty_sum_N + r_penalty * self.gamma ** i
            h_max_N = torch.where(h_max_N < hx, hx, h_max_N)
            # dis_ego_obs = torch.where(h_max_N < hx, hx, h_max_N)
            # dis_sum = dis_sum + dis_ego_obs

            c_sum_N = c_sum_N + c
            c_multi = c_multi * c
            live = live * (1 - dead)


        def Phi(y):
            m1 = 3/2
            m2 = m1 / (1 + m1) * 1
            m2 = 3/2
            tau = 0.2
            sig = (1 + tau * m1) / (1 + m2 * tau * torch.exp(torch.clamp(y / tau, min=-5, max=5)))

            # c = torch.relu(-y)
            return c

        self.sig_hx = Phi(h_max_N)

        self.r_sum_N = r_sum_N
        self.dis_sum = h_max_N


        self.c_sum_N = c_sum_N
        self.c_multi = c_multi
        self.death_rate = np.array(1 - live.mean(0))[:Config.num_rollout_agent]

        self.writer.add_scalar('train/r_N', (r_sum_N).mean().item(), self.iteration_index)
        self.writer.add_scalar('train/ r_tracking', r_tracking_sum_N.mean().item(), self.iteration_index)
        self.writer.add_scalar('train/ r_penalty', r_penalty_sum_N.mean().item(), self.iteration_index)
        self.writer.add_scalar('train/c_multi', (c_multi).mean().item(), self.iteration_index)
        self.writer.add_histogram('train/dis_sum', (h_max_N).mean().item(), self.iteration_index)
        self.writer.add_scalar('train/safe_prob', live.mean().item(), self.iteration_index)

        print('mol: live=%.4f dis_sum=%.4f return=%.1f' % (live.mean().detach().numpy(), h_max_N.mean().detach(), r_sum_N.mean().detach().numpy()))

    def accu_rew(self):
        self.ten_step_rew.insert(0, self.r_sum_N.mean().detach().item())
        if len(self.ten_step_rew) > 60:
            self.ten_step_rew.pop()
        # print(self.ten_step_rew)

    def PF(self):
        avg_rew_now = sum(self.ten_step_rew) / len(self.ten_step_rew)
        self.reward_min = self.reward_min if self.reward_min < avg_rew_now else avg_rew_now
        self.reward_max = self.reward_max if self.reward_max > avg_rew_now else avg_rew_now
        if avg_rew_now > 0.7 * self.reward_min:
            self.pf = 2 * self.pf
            self.pf = self.pf if self.pf < self.pf_max else self.pf_max
            self.reward_min = 0.0
        # if avg_rew_now < 1.6 * self.reward_max:
        #     self.pf = 0.8 * self.pf
        #     self.pf = self.pf if self.pf > self.pf_min else self.pf_min
        #     self.reward_max = -999.0
        # print(self.reward_min, self.reward_max, avg_rew_now, self.pf)
        self.writer.add_scalar('train/pf', self.pf, self.iteration_index)

    def PIM(self, ac):
        g_params = torch.autograd.grad(self.r_sum_N.mean() / 50, ac.pi.parameters(), retain_graph=True)
        obj_grad_ori = [g_param.contiguous() for g_param in g_params]
        g_vec_ori = nn.utils.convert_parameters.parameters_to_vector(obj_grad_ori)
        g_norm = torch.norm(g_vec_ori)
        g_vec_ori = g_vec_ori / max(g_norm, 1e-10) * min(1.5, g_norm)
        # g_vec = g_vec_ori / g_norm
        g_vec = g_vec_ori

        x0_vec = torch.zeros_like(g_vec)

        with torch.no_grad():
            action_old = ac.pi(self.obs)
        action = ac.pi(self.obs)
        policy_out_gain = torch.tensor([0.4, math.pi/2], dtype=torch.float32)  # 动作归一化系数
        grads = torch.autograd.grad(
            torch.mean(torch.pow((action - action_old) / policy_out_gain, 2)),
            ac.pi.parameters(), create_graph=True, retain_graph=True)
        policy_grad = [grad.contiguous() for grad in grads]  # 计算确定性动作kl散度(动作之差的2范数)，并求kl散度的一阶导数
        policy_grad_vec = nn.utils.convert_parameters.parameters_to_vector(policy_grad)


        def hvp(x: torch.Tensor):
            hvp_params = torch.autograd.grad(
                torch.sum(policy_grad_vec * x), ac.pi.parameters(), retain_graph=True)
            hvp_params = [hvp_param.contiguous() for hvp_param in hvp_params]
            return nn.utils.convert_parameters.parameters_to_vector(hvp_params)

        def cg_func(x: torch.Tensor):
            return hvp(x).add_(x, alpha=damping_factor)

        def _conjugate_gradient(
                Ax: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, x: torch.Tensor,
                rtol: float, atol: float, max_cg: int
        ):
            """Conjugate gradient method

            Solve $Ax=b$ where $A$ is a positive definite matrix.
            Refer to https://en.wikipedia.org/wiki/Conjugate_gradient_method.

            Args:
                Ax (Callable[[torch.Tensor], torch.Tensor]): Function to calculate $Ax$, return value shape (S,)
                b (torch.Tensor): b, shape (S,)
                x (torch.Tensor): Initial x value, shape (S,)
                rtol (float): Relative tolerance
                atol (float): Absolute tolerance
                max_cg (int): Maximum conjugate gradient iterations

            Raises:
                ValueError: When failed to converge within max_cg iterations

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Solution of $Ax=b$, residue
            """
            zero = x.new_zeros(())
            r = b - Ax(x)
            if torch.allclose(r, zero, rtol=rtol, atol=atol):
                print(f'mode0: {r.norm(2)} ?')
                return x, r

            r_dot = torch.dot(r, r)
            p = r.clone()
            for i in range(max_cg):
                Ap = Ax(p)
                alpha = r_dot / (torch.dot(p, Ap) + EPSILON)
                x = x.add_(p, alpha=alpha)
                r = r.add_(Ap, alpha=-alpha)
                if torch.allclose(r, zero, rtol=rtol, atol=atol):
                    print(f'mode1: {i} converged {r.norm(2)}')
                    return x, r
                r_dot, r_dot_old = torch.dot(r, r), r_dot
                beta = r_dot / r_dot_old
                p = r.add(p, alpha=beta)
            # print(f'mode2: max_cg {r.norm(2)}, {b.norm(2)}, {r.norm(2) / b.norm(2)}')
            return x, r

        x_vec, _ = _conjugate_gradient(cg_func, g_vec, x0_vec, rtol, atol, max_cg)

        cons_params = torch.autograd.grad(self.dis_sum.mean(), ac.pi.parameters(), retain_graph=True)
        cons_params = [cons_params.contiguous() for cons_params in cons_params]
        cons_vec_ori = nn.utils.convert_parameters.parameters_to_vector(cons_params)
        c_norm = torch.norm(cons_vec_ori)
        cons_vec_ori = cons_vec_ori / max(c_norm, 1e-10) * min(5, c_norm)
        # cons_vec = cons_vec_ori / c_norm
        cons_vec = cons_vec_ori

        weight_old = nn.utils.convert_parameters.parameters_to_vector(ac.pi.parameters())
        lr = 350e-4

        # b = (self.dis_sum.mean() - 0.04)
        # trust_region_part = torch.sqrt(2 * delta / (torch.dot(g_vec, x_vec) + EPSILON)) * x_vec
        # projection_part = torch.max(torch.tensor(0), (torch.dot(trust_region_part, cons_vec) + b) / (torch.dot(cons_vec, cons_vec) + EPSILON)) * cons_vec  # todo:trust region method

        # p_norm = torch.norm(projection_part)
        # t_norm = torch.norm(trust_region_part)
        # projection_part = projection_part / max(p_norm, 1e-10) * min(1.0*t_norm, p_norm)

        # projection_coeff = torch.max(torch.tensor(0), (torch.dot(g_vec, cons_vec) + b) / (torch.dot(cons_vec, cons_vec) + EPSILON))
        # projection_coeff = torch.where(projection_coeff < torch.tensor([10.0]), projection_coeff, torch.tensor([10.0]))
        # projection_part = projection_coeff * cons_vec

        # print('投影是否大于0---', (torch.dot(trust_region_part, cons_vec) + b) / (torch.dot(cons_vec, cons_vec) + EPSILON).detach(), self.dis_sum.mean().detach())
        # print('目标函数梯度范数 vs 约束梯度范数 vs trpo范数 vs proj范数 vs 旧策略范数---', g_norm, c_norm, torch.norm(trust_region_part), torch.norm(projection_part), torch.norm(weight_old))
        # print('dot和b的关系---', torch.dot(g_vec, cons_vec), b.detach())


        # padp_step = trust_region_part - projection_part
        step = g_vec - self.pf*cons_vec


        weight_new = weight_old.add(lr*step)
        nn.utils.convert_parameters.vector_to_parameters(weight_new, ac.pi.parameters())
        action_new = ac.pi(self.obs)
        one_step_kl = torch.mean(torch.pow((action_new - action_old) / policy_out_gain, 2))

        

        self.writer.add_scalar('kl_divergence_padp', one_step_kl, self.iteration_index)
        # self.writer.add_scalar('train/trpo_norm', torch.norm(trust_region_part), self.iteration_index)
        # self.writer.add_scalar('train/projection_norm', torch.norm(projection_part), self.iteration_index)
        self.writer.add_scalar('train/cons_norm', torch.norm(cons_vec_ori), self.iteration_index)
        self.writer.add_scalar('train/obj_norm', torch.norm(g_vec_ori), self.iteration_index)
        # self.writer.add_scalar('dot_norm', torch.norm(torch.dot(trust_region_part, cons_vec)), self.iteration_index)
        # self.writer.add_scalar('train/projection_coeff', torch.norm(projection_coeff), self.iteration_index)

    def next_state(self, ac, state_model):
        with torch.no_grad():
            x_init = state_model.reset()
            for i in range(self.num_agent):
                if self.init_index[i] > 0:
                    self.x[i] = x_init[i]
                    self.init_index[i] = 0
                    self.step[i] = 0

            u = ac.step(self.x)
            # if torch.rand(1)>0.9:
            #     u = torch.Tensor(np.random.normal([0, 0], [0.4, 0.7], [x_init.shape[0], 2])) # todo:注释动作随机性

            x_next, _, _, dead, done, _ = state_model.step(self.x, u)
            self.x_next = x_next.detach()
            next_buffer_index = self.buffer_index+self.num_agent
            if next_buffer_index<self.num_buffer:
                self.x_buffer[self.buffer_index:next_buffer_index] = self.x_next
                self.buffer_index = next_buffer_index
            else:
                self.x_buffer[self.buffer_index:self.num_buffer] = self.x_next[self.buffer_index-self.num_buffer]
                self.x_buffer[:next_buffer_index-self.num_buffer]
                self.buffer_index = next_buffer_index-self.num_buffer
                self.buffer_full = True


            self.step = self.step + 1
            self.init_index = 0*dead.sum(1).reshape(-1, 1) + (self.step>self.num_max_step) + done.reshape(-1, 1)
