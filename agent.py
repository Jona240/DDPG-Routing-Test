"""
参考https://www.cnblogs.com/pinard/p/10345762.html
"""
import torch
from net import ActionNet, CriticNet
from net import ActorNetLoss, CriticNetLoss


class Agent:
    def __init__(self):
        # 基本参数
        self.gamma = 0.8  # 衰减系数
        self.tau_actor, self.tau_critic = 0.1, 0.1  # 软更新系数
        self.batch_size = 32
        self.actor_updata_freq = 4  # 动作网络更新系数
        self.critic_updata_freq = 4  # 评论网络更新频率
        self.noise = None
        # 动作网络
        self.actor_net = ActionNet()
        self.actor_net_now = ActionNet()
        self.actor_net_loss = ActorNetLoss()
        self.actor_optimizer = None
        self.theta = []
        self.theta_now = []
        # 评论网络
        self.critic_net = CriticNet()
        self.critic_net_now = CriticNet()
        self.critic_net_loss = CriticNetLoss()
        self.critic_optimizer = None
        self.omega = []
        self.omega_now = []

        # 经验回放集合
        self.experience_playback = {'state': [],
                                    'action': [],
                                    'next_state': [],
                                    'reward': [],
                                    'is_terminal': []}

    @staticmethod
    def update_now_net2target_net(model_now, model, soft_update):
        model_now_sd = model_now.state_dict()
        model_sd = model.state_dict()
        su = soft_update
        model_update_sd = {layer: model_now_sd[layer] * su + model_sd[layer] * (1 - su) for layer in model_sd}
        model.load_state_dict(model_update_sd)


    def critic_learn(self, batch_size, retain_graph=False, gpu_list=''):
        """当前评论网络参数更新"""
        if len(self.experience_playback['state']) < batch_size:
            batch_size = len(self.experience_playback['state'])
        ep_state_b = torch.cat(tuple(self.experience_playback['state'][-batch_size:]), 0)
        ep_next_state_b = torch.cat(tuple(self.experience_playback['next_state'][-batch_size:]), 0)
        next_action_b = self.actor_net(ep_next_state_b)
        ep_action_b = torch.cat(tuple(self.experience_playback['action'][-batch_size:]), 0)
        r_b = torch.cat(tuple(self.experience_playback['reward'][-batch_size:]), 0)
        ist_b = torch.cat(tuple(self.experience_playback['is_terminal'][-batch_size:]), 0)
        next_state_action_b = torch.cat((ep_next_state_b, next_action_b), 1)
        next_q = self.critic_net(next_state_action_b)
        gamma_b = torch.Tensor([self.gamma]).repeat(batch_size)
        if not gpu_list == '':
            gamma_b = gamma_b.cuda()
            ist_b = ist_b.cuda()
            r_b = r_b.cuda()
        # q = r_b + ist_b * next_q * gamma_b
        q = ist_b * next_q
        q = q * gamma_b
        q = q + r_b
        ep_state_action_b = torch.cat((ep_state_b, ep_action_b), 1)
        self.critic_optimizer.zero_grad()
        q = q.detach()
        ep_state_action_b = ep_state_action_b.detach()
        if not gpu_list == '':
            q = q.cuda()
            ep_state_action_b = ep_state_action_b.cuda()
        critic_q = self.critic_net_now(ep_state_action_b)
        critic_loss = self.critic_net_loss(critic_q, q)
        critic_loss.backward(retain_graph=retain_graph)
        self.critic_optimizer.step()
        return critic_loss


    def actor_learn(self, batch_size, retain_graph=False, gpu_list=''):
        """actor当前网络参数更新"""
        ep_state_b = torch.cat(tuple(self.experience_playback['state'][-batch_size:]), 0)
        # ep_action_b = torch.cat(tuple(self.experience_playback['action'][-batch_size:]), 0)
        # ep_state_action_b = torch.cat((ep_state_b, ep_action_b), 1)
        self.actor_optimizer.zero_grad()
        ep_action_b = self.actor_net_now(ep_state_b)
        ep_state_action_b = torch.cat((ep_state_b, ep_action_b), 1)
        critic_q = self.critic_net_now(ep_state_action_b)
        batch_size_tensor = torch.Tensor([batch_size])
        if gpu_list == '':
            batch_size_tensor = batch_size_tensor.cuda()
        actor_loss = self.actor_net_loss(critic_q)
        actor_loss.backward(retain_graph=retain_graph)
        self.actor_optimizer.step()
        return actor_loss

    @staticmethod
    def state2actor_tensor(state, gpu_list):
        l_p = state[1]
        cm = state[2]
        cm_w = state[3]
        x = []
        cm = cm.reshape(1, -1)[0]
        [x.append(temp) for temp in cm]
        cm_w = cm_w.reshape(1, -1)[0]
        [x.append(temp) for temp in cm_w]
        x.append(l_p)
        x = torch.Tensor(x).unsqueeze(0)
        if not gpu_list == '':
            x = x.cuda()
        return x


if __name__ == "__main__":
    pass
