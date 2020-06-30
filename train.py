from agent import Agent
from environment import SWEnv
import networkx as nx
import logging
import datetime
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import json


def get_logger(log_dir, file_filter="DEBUG", name=__name__, on_screen=True):
    """
    get logger
    :param name: log name
    :param log_dir: log dir
    :param file_filter: DEBUG or INFO
    :param on_screen : should print log？ T or F
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, name))
    assert (file_filter == "DEBUG" or file_filter == "INFO")
    if file_filter == "DEBUG":
        file_handler.setLevel(logging.DEBUG)
    elif file_filter == "INFO":
        file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if on_screen:  # print log on console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def text_create(name, msg):     # 创建一个txt
    full_path = name + '.txt'
    file = open(full_path, 'w')
    file.write(msg)             # 写入部分msg
    file.close()

def Write_Text(file_name,contant):  #写入文档
    # file_name = 'test.txt'
    with open(file_name,"a+") as f:
        f.writelines(contant)
        f.writelines("\n")

def main(iteration_num=2,
         iteration_graph=50,
         checkpoint_freq=1,
         gpu_list='0',
         log_dir=None,
         actor_pretrain='',
         critic_pretrain=''):
    """GET TIME"""
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    """CREATE DIR"""
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    """BEGIN LOGGING"""
    logger = get_logger(log_dir, name="agent_0")
    text_create(str(log_dir) + 'Reward','Reward:\n')
    text_create(str(log_dir) + 'Path', 'Path:\n')
    text_create(str(log_dir) + 'Save model', 'Save model:\n')
    text_create(str(log_dir) + 'indexes', 'Indexes:\n')
    logger.info("EXPERIMENT TIME:" + timestr)
    # 创建环境
    env = SWEnv()
    # 创建个体
    agent = Agent()
    lead_rate = 0           # 引导概率
    boost_rate = 0.02       # 引导加速概率
    # 初始化网络
    if not gpu_list == '':
        agent.actor_net = agent.actor_net.cuda()
        agent.critic_net = agent.critic_net.cuda()
        agent.actor_net_now = agent.actor_net_now.cuda()
        agent.critic_net_now = agent.critic_net_now.cuda()
        agent.actor_net_loss.cuda()
        agent.critic_net_loss.cuda()
    if not actor_pretrain == '':
        agent.actor_net.load_state_dict(torch.load(actor_pretrain))
        agent.actor_net_now.load_state_dict(torch.load(actor_pretrain))
    if not critic_pretrain == '':
        agent.critic_net.load_state_dict(torch.load(actor_pretrain))
        agent.critic_net_now.load_state_dict(torch.load(actor_pretrain))
    agent.actor_net.train()
    agent.critic_net.train()
    agent.actor_net_now.train()
    agent.critic_net_now.train()#.train
    # 优化器
    agent.actor_optimizer = torch.optim.Adam(
        agent.actor_net_now.parameters(),
        lr=0.001,  # 学习率
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    agent.critic_optimizer = torch.optim.Adam(
        agent.critic_net_now.parameters(),
        lr=0.001,  # 学习率
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    # 学习率衰减
    actor_scheduler = torch.optim.lr_scheduler.StepLR(agent.actor_optimizer, step_size=5, gamma=0.9)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(agent.critic_optimizer, step_size=5, gamma=0.9)

    for it in range(iteration_num):
        # 生成图
        logger.info("-----------------------------------")
        logger.info("iteration_num:",it)
        env.change_graph()
        cl_list = []                                # 记录两个loss
        al_list = []
        actor_scheduler.step()                      # 进行学习率衰减
        critic_scheduler.step()

        target_graph = env.ws
        ps = nx.circular_layout(target_graph)  # 布置框架
        nx.draw(target_graph, ps, with_labels=False, node_size=30)
        plt.savefig(str(log_dir) + 'train_Graph %d' % it)
        plt.close()


        G = nx.Graph()  # 小世界网络转出至另一个图中
        for i in range(30):  # 添加边
            for j in range(i, 30):
                if env.cm[i, j] != 0:
                    G.add_edge(i, j, weight=env.cm_weight[i, j])

        Write_Text(str(log_dir) + 'Path.txt', 'train_graph No.%d ' % it)


        for itg in tqdm(range(iteration_graph), total=iteration_graph, smoothing=0.9):
            # 重置环境
            state = env.reset()
            local_position = state[1]
            state = agent.state2actor_tensor(state,gpu_list)  # 转化为tensor
            if gpu_list == '':
                state = state.cuda()
            is_terminal = False

            env.indexes_update()  # 浮动指标并写入日志

            Write_Text(str(log_dir) + 'indexes.txt', '\nTime_%d' % it)
            Write_Text(str(log_dir) + 'indexes.txt', 'Bandwidth:')
            for i in range(30):
                Write_Text(str(log_dir) + 'indexes.txt', '%.2f ' % env.bandwidth[i])
            Write_Text(str(log_dir) + 'indexes.txt', 'occupy_ratio:')
            for i in range(30):
                Write_Text(str(log_dir) + 'indexes.txt', '%.4f ' % env.occupy_ratio[i])
            Write_Text(str(log_dir) + 'indexes.txt', 'time_shake:')
            for i in range(30):
                Write_Text(str(log_dir) + 'indexes.txt', '%.2f ' % env.time_shake[i])
            Write_Text(str(log_dir) + 'indexes.txt', 'transmit_ratio:')
            for i in range(30):
                Write_Text(str(log_dir) + 'indexes.txt', '%.4f ' % env.transmit_ratio[i])
            Write_Text(str(log_dir) + 'indexes.txt', 'info_accord:')
            for i in range(30):
                Write_Text(str(log_dir) + 'indexes.txt', '%.4f ' % env.info_accord[i])

            # 清空ep
            agent.experience_playback = {'state': [],
                                         'action': [],
                                         'reward': [],
                                         'next_state': [],
                                         'is_terminal': []}

            Write_Text(str(log_dir) + 'Path.txt', 'Route No.%d ' % itg)
            total_r = 0
            # 迭代训练
            while not is_terminal:
                # 动作网络
                a_output = agent.actor_net_now(state)
                # 执行动作
                a = a_output.data.max(1)[1]
                # 当下Agent所做的选择
                present_step = int(a[0])
                action_lead_p = random.random()
                if action_lead_p < lead_rate:
                    path = nx.dijkstra_path(G, source=local_position, target=env.terminal)
                    present_step = path[1]
                    local_position = path[1]
                lead_rate = lead_rate + boost_rate

                next_state, r, is_terminal, _ = env.step(present_step)
                Write_Text(str(log_dir) + 'Path.txt', '%d' % present_step)
                total_r = total_r + r
                next_state = agent.state2actor_tensor(next_state,gpu_list)
                r = torch.Tensor([r])

                if is_terminal:
                    is_terminal_bit = torch.Tensor([0])
                else:
                    is_terminal_bit = torch.Tensor([1])
                if not gpu_list == '':
                    next_state = next_state.cuda()
                    r = r.cuda()
                    is_terminal_bit = is_terminal_bit.cuda()
                # 经验回放
                agent.experience_playback['state'].append(state)
                agent.experience_playback['action'].append(a_output)
                agent.experience_playback['reward'].append(r)
                agent.experience_playback['next_state'].append(next_state)
                agent.experience_playback['is_terminal'].append(is_terminal_bit)
                state = next_state
                # 更新当前critic网络
                cl=agent.critic_learn(agent.batch_size, gpu_list=gpu_list)
                # 更新当前actor网络
                al=agent.actor_learn(agent.batch_size, gpu_list=gpu_list)
                cl_list.append(cl.cpu().detach().numpy().tolist())
                al_list.append(al.cpu().detach().numpy().tolist())

            Write_Text(str(log_dir) + 'Path.txt', '本次总Reward：%d' % total_r)
            Write_Text(str(log_dir) + 'Path.txt', '\n')
            # 目标网络参数更新
            if itg % agent.actor_updata_freq == 1:  # 目标动作网络更新
                agent.update_now_net2target_net(agent.actor_net_now, agent.actor_net, agent.tau_actor)
            if itg % agent.critic_updata_freq == 1:  # 目标动作网络更新
                agent.update_now_net2target_net(agent.critic_net_now, agent.critic_net, agent.tau_critic)
        logger.info('iter num:'+str(it)+'----'+'actor loss:'+str(al_list))
        logger.info('iter num:'+str(it)+'----'+'critic loss:'+str(cl_list))
        # 测试代码
        with torch.no_grad():
            # 重置环境
            state = env.reset()
            print("连接图")
            print(state[3])
            state = agent.state2actor_tensor(state,gpu_list)  # 转化为tensor
            if gpu_list == '':
                state = state.cuda()
            is_terminal = False
            # 清空ep
            agent.experience_playback = {'state': [],
                                         'action': [],
                                         'reward': [],
                                         'next_state': [],
                                         'is_terminal': []}
            while not is_terminal:
                # 动作网络
                a_output = agent.actor_net_now(state)
                # 执行动作
                a = a_output.data.max(1)[1]
                next_state, r, is_terminal, _ = env.step(int(a[0]))
                next_state = agent.state2actor_tensor(next_state,gpu_list)
                r = torch.Tensor([r])

                if is_terminal:
                    is_terminal_bit = torch.Tensor([0])
                else:
                    is_terminal_bit = torch.Tensor([1])
                if gpu_list == '':
                    next_state = next_state.cuda()
                    r = r.cuda()
                    is_terminal_bit = is_terminal_bit.cuda()
                # 经验回放
                agent.experience_playback['state'].append(state)
                agent.experience_playback['action'].append(a_output)
                agent.experience_playback['reward'].append(r)
                agent.experience_playback['next_state'].append(next_state)
                agent.experience_playback['is_terminal'].append(is_terminal_bit)
                state = next_state

            finial_reward=torch.cat(tuple(agent.experience_playback['reward']),dim=0)
            logger.info("test reward:",torch.sum(finial_reward,0))
            logger.debug("path:",agent.experience_playback['action'])
        # 网络参数保存
        if it % checkpoint_freq == 0:
            logger.info('Save model...')
            Write_Text(str(log_dir) + 'Save model.txt', 'Save model...\n')
            actor_savepath = str(checkpoints_dir) + '\\actor_model_iter' + str(it) + '.pth'
            critic_savepath = str(checkpoints_dir) + '\\critic_model_iter' + str(it) + '.pth'
            logger.info('Actor Net Saving at %s' % actor_savepath)
            logger.info('Critic Net Saving at %s' % critic_savepath)
            Write_Text(str(log_dir) + 'Save model.txt', 'Actor Net Saving at %s\n' % actor_savepath)
            Write_Text(str(log_dir) + 'Save model.txt', 'Critic Net Saving at %s\n' % critic_savepath)
            torch.save(agent.actor_net.state_dict(), actor_savepath)
            torch.save(agent.critic_net.state_dict(), critic_savepath)


if __name__ == '__main__':
    main()
