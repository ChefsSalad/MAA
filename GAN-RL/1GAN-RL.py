# 三路交叉对抗D3QN架构（统一输入版本）
# 生成器生成状态粒子和动作价值，判别器评估状态分布
# 输入：合并后的特征（价格+技术+基本面）

import os
import sys
import json
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ==================== 配置部分 ====================
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 超参数配置
SEQ_LENGTH = 10        # 时序窗口长度
PRICE_FEATURES = 4     # OHLC特征
INPUT_DIM = PRICE_FEATURES
ACTION_DIM = 3         # 动作空间：买入(0)、卖出(1)、持有(2)
HIDDEN_DIM = 128       # 隐藏层维度
STATE_PARTICLES = 30   # 状态粒子数量
BATCH_SIZE = 64        
LR_GENERATOR = 0.001
LR_DISCRIMINATOR = 0.001
GAMMA = 0.9 
MEMORY_SIZE = 10000
GRADIENT_PENALTY = 5  # WGAN-GP梯度惩罚系数

# 路径配置
MODEL_DIR = "saved_models"
METRICS_DIR = "training_metrics"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ==================== 模型组件 ====================
class GRUGenerator(nn.Module):
    """FC生成器（保留单生成器结构）"""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * SEQ_LENGTH, HIDDEN_DIM * 2),
            nn.LayerNorm(HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2) 
        )
        self.state_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, STATE_PARTICLES*PRICE_FEATURES)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        fc_out = self.fc(x)
        state_particles = self.state_head(fc_out).view(-1, STATE_PARTICLES, PRICE_FEATURES)
        V = self.value_stream(fc_out)
        A = self.advantage_stream(fc_out)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q, state_particles, V, A

class Discriminator(nn.Module):
    """Wasserstein判别器"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PRICE_FEATURES, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)
# ==================== 智能体与训练逻辑 ==================== 
class StockTradingAgent:
    def __init__(self):
        # 单生成器+单判别器结构
        self.generator = GRUGenerator(INPUT_DIM, ACTION_DIM)
        self.target_generator = GRUGenerator(INPUT_DIM, ACTION_DIM)
        self.target_generator.load_state_dict(self.generator.state_dict())
        self.target_generator.eval()
        
        self.discriminator = Discriminator()
        self.opt_g = optim.Adam(self.generator.parameters(), lr=LR_GENERATOR)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=LR_DISCRIMINATOR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.manager = ModelManager(self, None)
        self._init_training_history()

    def _init_training_history(self):
        """初始化训练历史记录"""
        self.training_history = {
            'rewards': [],
            'd_losses': [],
            'g_losses': [],
            'dqn_losses': [],
            'adv_losses': [],
            'grad_norms': [],
            'action_dist': np.zeros(ACTION_DIM)
        }

    def _gradient_penalty(self, real_samples, fake_samples, D):
        """计算WGAN-GP梯度惩罚"""
        alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def _update_discriminator(self, real_states_batch):
        """更新判别器（单判别器版本）"""
        real_price_data = real_states_batch[:, :, :PRICE_FEATURES]
        real_price_flattened = real_price_data.reshape(-1, PRICE_FEATURES)
        
        # 生成器生成假样本
        with torch.no_grad():
            _, states, _, _ = self.generator(real_states_batch)
        fake_states = states.reshape(-1, PRICE_FEATURES)
        
        # 平衡样本数量
        num_real = real_price_flattened.size(0)
        num_fake = fake_states.size(0)
        if num_fake > num_real:
            fake_states = fake_states[:num_real]
        elif num_fake < num_real:
            fake_states = fake_states.repeat((num_real//num_fake)+1, 1)[:num_real]
        
        # 计算判别损失
        real_validity = self.discriminator(real_price_flattened)
        fake_validity = self.discriminator(fake_states.detach())
        gp = self._gradient_penalty(real_price_flattened, fake_states, self.discriminator)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + GRADIENT_PENALTY * gp
        
        # 反向传播
        self.opt_d.zero_grad()
        d_loss.backward()
        self.opt_d.step()
        return d_loss.item()

    def _update_generator(self, states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch):
        """更新生成器（单生成器版本）"""
        states = torch.FloatTensor(np.array(states_batch))
        next_states = torch.FloatTensor(np.array(next_states_batch))
        actions = torch.LongTensor(actions_batch)
        rewards = torch.FloatTensor(rewards_batch)
        dones = torch.FloatTensor(dones_batch)

        # DQN损失
        current_q, _, _, _ = self.generator(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q, _, _, _ = self.target_generator(next_states)
            target_q = rewards + (1 - dones) * GAMMA * next_q.max(1)[0].unsqueeze(1)
        
        dqn_loss = F.mse_loss(current_q, target_q)
        
        # 对抗损失
        _, gen_states, _, _ = self.generator(states)
        gen_states = gen_states.view(-1, PRICE_FEATURES)
        adv_loss = -self.discriminator(gen_states).mean()
        
        # 总损失
        total_loss = dqn_loss + 0.2 * adv_loss
        
        # 反向传播
        self.opt_g.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 2.0)
        self.opt_g.step()
        
        # 记录梯度
        grad_norms = [p.grad.norm().item() for p in self.generator.parameters() if p.grad is not None]
        self.training_history['grad_norms'].append(np.mean(grad_norms) if grad_norms else 0)
        
        return dqn_loss.item(), adv_loss.item()

    def soft_update_targets(self, tau=0.005):
        """软更新目标网络"""
        for t_param, o_param in zip(self.target_generator.parameters(), self.generator.parameters()):
            t_param.data.copy_(tau * o_param.data + (1 - tau) * t_param.data)

    def get_action(self, state, epsilon=0.2):
        """动作选择（单生成器版本）"""
        if np.random.rand() < epsilon:
            return np.random.randint(ACTION_DIM)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values, _, _, _ = self.generator(state_tensor)
            action = q_values.argmax().item()
            self.training_history['action_dist'][action] += 1
            return action

    def update(self):
        """统一更新入口"""
        if len(self.memory) < BATCH_SIZE:
            return 0, (0, 0)
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
        
        # 更新判别器
        d_loss = self._update_discriminator(torch.FloatTensor(np.array(states_b)))
        
        # 更新生成器
        dqn_loss, adv_loss = self._update_generator(states_b, actions_b, rewards_b, next_states_b, dones_b)
        self.soft_update_targets()
        
        return d_loss, (dqn_loss, adv_loss)

class ModelManager:
    """模型管理组件"""
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.best_reward = -np.inf
        self.best_metrics = {}
        self.history = {'train': [], 'val': [], 'test': []}

    def save_checkpoint(self, episode, is_best=False, suffix=''):
        """保存训练检查点"""
        checkpoint = {
            'episode': episode,
            'generator': self.agent.generator.state_dict(),
            'discriminator': self.agent.discriminator.state_dict(),
            'opt_g': self.agent.opt_g.state_dict(),
            'opt_d': self.agent.opt_d.state_dict(),
            'training_history': self.agent.training_history,
            'best_reward': self.best_reward,
            'env_config': {
                'seq_length': self.env.seq_length,
                'position': self.env.position
            }
        }
        filename = f'checkpoint_ep{episode}{suffix}.pth'
        torch.save(checkpoint, os.path.join(MODEL_DIR, filename))
        if is_best:
            torch.save(checkpoint, os.path.join(MODEL_DIR, 'model_best.pth'))

    def load_checkpoint(self, path, resume_training=True):
        """加载训练检查点"""
        checkpoint = torch.load(path)
        self.agent.generator.load_state_dict(checkpoint['generator'])
        self.agent.discriminator.load_state_dict(checkpoint['discriminator'])
        self.agent.opt_g.load_state_dict(checkpoint['opt_g'])
        self.agent.opt_d.load_state_dict(checkpoint['opt_d'])
        if resume_training:
            self.agent.training_history = checkpoint['training_history']
            self.best_reward = checkpoint['best_reward']
            logger.info(f"恢复训练状态，最佳奖励：{self.best_reward:.2f}")

    def evaluate(self, n_episodes=10, epsilon=0.0):
        """增强评估指标"""
        metrics = {
            'total_returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'action_dist': np.zeros(ACTION_DIM),
            'steps':[]
        }

        for _ in range(n_episodes):
            state = self.env.reset()  # 使用固定起始点
            done = False
            portfolio_history = [self.env.init_balance]
            step_history = []  # 用来记录每一步的详细数据

            while not done:
                action = self.agent.get_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                portfolio_history.append(self.env.balance + self.env.holdings * self.env.combined_data[self.env.current_step-1, 3])
                metrics['action_dist'][action] += 1
                state = next_state
                step_data = {
                    'step': self.env.current_step,
                    'action': action,
                    'balance': self.env.balance,
                    'holdings': self.env.holdings,
                    'portfolio_value': self.env.balance + self.env.holdings * self.env.combined_data[self.env.current_step-1, 3]
                }
                step_history.append(step_data)  # 记录每一步的数据

            # 计算各项指标
            returns = (portfolio_history[-1] - self.env.init_balance) / self.env.init_balance
            metrics['total_returns'].append(returns)
            
            returns_series = np.diff(portfolio_history) / portfolio_history[:-1]
            sharpe = np.mean(returns_series) / (np.std(returns_series) + 1e-9)
            metrics['sharpe_ratios'].append(sharpe)
            
            peak = np.maximum.accumulate(portfolio_history)
            drawdown = (peak - portfolio_history) / peak
            metrics['max_drawdowns'].append(np.max(drawdown))

            # 将每个回合的步骤信息存储到文件中
            with open(f'step_history_episode{_}.json', 'w') as f:
                json.dump(step_history, f)

            # 将步骤记录存入到metrics字典中
            metrics['steps'].append(step_history)  # 存储每个回合的所有步骤

        return {
            'mean_return': np.mean(metrics['total_returns']),
            'std_return': np.std(metrics['total_returns']),
            'sharpe_ratio': np.mean(metrics['sharpe_ratios']),
            'max_drawdown': np.mean(metrics['max_drawdowns']),
            'action_dist': metrics['action_dist'].tolist(),
            'steps': metrics['steps']  # 返回每个回合的步骤信息
        }

    def _calculate_drawdown(self, portfolio_values):
        """计算最大回撤"""
        max_drawdowns = []
        for pv in portfolio_values:
            peak = np.maximum.accumulate(pv)
            trough = np.minimum.accumulate(pv)
            max_drawdowns.append(np.max((peak - trough)/peak))
        return np.mean(max_drawdowns)

# ==================== 数据与环境 ====================
def load_and_process_data():

    price_df = pd.read_csv('Soybean.csv', parse_dates=['time'])
    price_df = price_df[['time', 'open', 'high', 'low', 'close']].set_index('time')
    price_df = price_df[['open', 'high', 'low', 'close']].values.astype(np.float32)
        
    return price_df



class StockTradingEnv:
    """股票交易环境"""
    def __init__(self, combined_data, seq_length=10):
        self.combined_data = combined_data
        # self.price_data = price_data
        self.seq_length = seq_length
        self.current_step = seq_length
        self.position = 0.0

        self.transFee = 100  # 添加交易手续费
        self.init_balance = 500000  # 初始资金

        # 噪声配置参数
        self.noise_level = 0.01  # 初始噪声水平
        self.noise_type = 'gaussian'  # 可选 'uniform'
        self.dynamic_noise = True  # 是否启用动态噪声调整

    def _add_noise(self, state):
        """为输入状态添加可控噪声"""
        if self.noise_type == 'gaussian':
            noise = np.random.normal(
                scale=self.noise_level * np.std(state, axis=0),
                size=state.shape
            )
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(
                low=-self.noise_level,
                high=self.noise_level,
                size=state.shape
            )
        else:
            raise ValueError(f"未知噪声类型: {self.noise_type}")
        
        # 应用噪声并确保数据有效性
        noisy_state = state + noise
        noisy_state[:, :3] = np.clip(noisy_state[:, :3], a_min=0, a_max=None)  # OHL不能为负
        return noisy_state.astype(np.float32)
    
    def _update_noise_level(self, episode):
        """动态调整噪声水平（在训练循环中调用）"""
        if self.dynamic_noise:
            # 随训练进程线性衰减噪声
            self.noise_level = max(0.001, 0.05 * (1 - episode/1000)) 
    
    def reset(self):
        self.balance = self.init_balance
        self.holdings = 0
        self.portfolio_values = []
        
        self.current_step = self.seq_length
        self.position = 0
        return self._get_state()

    def step(self, action):
        prev_close = self.combined_data[self.current_step-2, 3]  
        current_close = self.combined_data[self.current_step-1, 3]
        reward_k = 0

        # 有手续费版
        # if action == 0:  # 买入开多仓
        #     if self.balance > self.transFee:
        #         max_afford = (self.balance - self.transFee) / current_close
        #         self.holdings += max_afford
        #         self.balance = 0  # 全部余额用于买入
        # elif action == 1:  # 卖出开空仓
        #     if self.balance > self.transFee:
        #         max_afford = (self.balance - self.transFee) / current_close
        #         self.holdings -= max_afford  # 持仓变为负数表示空头
        #         self.balance += max_afford * current_close - self.transFee  # 卖出获得资金
        # elif action == 2:  # 平仓
        #     if self.holdings > 0:  # 平多仓
        #         sell_value = self.holdings * current_close
        #         if sell_value > self.transFee:
        #             self.balance += sell_value - self.transFee
        #             self.holdings = 0
        #     elif self.holdings < 0:  # 平空仓
        #         required_cash = abs(self.holdings) * current_close + self.transFee
        #         if self.balance >= required_cash:
        #             self.balance -= required_cash
        #             self.holdings = 0

        if action == 0:  # 买入开多仓
            if self.balance > 0:
                max_afford = (self.balance - 0) / current_close
                self.holdings += max_afford
                self.balance = 0  # 全部余额用于买入
        elif action == 1:  # 卖出开空仓
            if self.balance > 0:
                max_afford = (self.balance - 0) / current_close
                self.holdings -= max_afford  # 持仓变为负数表示空头
                self.balance += max_afford * current_close - 0  # 卖出获得资金
        elif action == 2:  # 平仓
            if self.holdings > 0:  # 平多仓
                sell_value = self.holdings * current_close
                if sell_value > 0:
                    self.balance += sell_value - 0
                    self.holdings = 0
            elif self.holdings < 0:  # 平空仓
                required_cash = abs(self.holdings) * current_close + 0
                if self.balance >= required_cash:
                    self.balance -= required_cash
                    self.holdings = 0



        # 计算新奖励函数
        current_value = self.balance + self.holdings * current_close
        self.portfolio_values.append(current_value)
        reward = self._calculate_reward(current_value)  +  5 * reward_k

        self.current_step += 1  # 移动到下一个时间步
        done = self.current_step >= len(self.combined_data) - 1
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, current_value):
        """复合奖励函数"""
        # 1. 收益率奖励
        if len(self.portfolio_values) > 1:
            returns = (current_value - self.init_balance) / 50000
        else:
            returns = 0.0


        # # 2. 夏普率奖励（滑动窗口）
        # window_size = 20
        # if len(self.portfolio_values) > window_size:
        #     returns_series = np.diff(self.portfolio_values[-window_size:]) / self.init_balance
        #     sharpe = np.mean(returns_series) / (np.std(returns_series) + 1e-9)
        #     sharpe_reward = sharpe * 0.1
        # else:
        #     sharpe_reward = 0

        # # 3. 回撤惩罚
        # peak = np.max(self.portfolio_values)
        # drawdown = (peak - current_value) / peak if peak > 0 else 0
        # drawdown_penalty = -drawdown * 0.5

        return returns
        # return returns + sharpe_reward + drawdown_penalty

    def _get_state(self):
        raw_state = self.combined_data[self.current_step-self.seq_length : self.current_step]
        return self._add_noise(raw_state)

# ==================== 训练与测试 ====================
def plot_training_metrics(agent, episode, val_metrics=None):
    """绘制训练指标"""
    plt.figure(figsize=(24, 16))
    sns.set_style("whitegrid")
    plt.suptitle(f'Training Metrics at Episode {episode}', y=1.02)

    metrics = [
        ('rewards', 'Episode Rewards'),
        ('d_losses', 'Discriminator Loss'),
        ('g_losses', 'Generator Loss'),
        ('dqn_losses', 'DQN Loss'),
        ('adv_losses', 'Adversarial Loss'),
        ('grad_norms', 'Gradient Norms')
    ]
    
    for i, (key, title) in enumerate(metrics[:6], 1):
        plt.subplot(3, 4, i)
        plt.plot(agent.training_history[key])
        plt.title(title)
        if i == 1:
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.subplot(3, 4, 6)
    actions = ['Buy', 'Sell', 'Hold']
    plt.bar(actions, agent.training_history['action_dist'], color=['green', 'red', 'blue'])
    plt.title('Action Distribution')

    if val_metrics:
        plt.subplot(3, 4, 7)
        plt.bar(['Return', 'Sharpe', 'Drawdown'], 
                [val_metrics['mean_return'], val_metrics['sharpe_ratio'], val_metrics['max_drawdown']],
                color=['green', 'blue', 'red'])
        plt.title('Validation Metrics')

    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, f'metrics_ep{episode}.png'))
    plt.close()

def train():
    """训练主循环"""
    combined_data = load_and_process_data()
    env = StockTradingEnv(combined_data)
    agent = StockTradingAgent()
    agent.manager.env = env  # 注入环境实例

    for episode in range(1000):
        env._update_noise_level(episode)  # 更新噪声水平
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.append((state.copy(), action, reward, next_state.copy(), done))
            state = next_state
            total_reward += reward

            if len(agent.memory) >= BATCH_SIZE:
                d_loss, (dqn_loss, adv_loss) = agent.update()
                if len(agent.memory) % 10 == 0:
                    recent_memory = list(agent.memory)[-BATCH_SIZE:]
                    states_tensor = torch.FloatTensor(np.array([s[0] for s in recent_memory]))
                    q_values = agent.generator(states_tensor)[0].detach().cpu().numpy()
                    logger.info(
                        f"Q Value Stats: Mean={np.mean(q_values):.2f}±{np.std(q_values):.2f} | "  # 修改后
                        f"d_loss: {d_loss:.2f} | "
                        f"dqn_loss: {dqn_loss:.2f} | "  # 缺少分隔符
                        f"adv_loss: {adv_loss:.2f}"  # 修改后
                    )

        # 更新训练历史
        agent.training_history['rewards'].append(total_reward)
        agent.training_history['d_losses'].append(d_loss)
        agent.training_history['g_losses'].append(dqn_loss + adv_loss)
        agent.training_history['dqn_losses'].append(dqn_loss)
        agent.training_history['adv_losses'].append(adv_loss)

        # 定期保存与评估
        if episode % 3 == 0:
            agent.manager.save_checkpoint(episode)
            val_metrics = agent.manager.evaluate(n_episodes=5)
            logger.info(f"验证指标：{json.dumps({
                'mean_return': float(val_metrics['mean_return']),
                'std_return': float(val_metrics['std_return']),
                'sharpe_ratio': float(val_metrics['sharpe_ratio']),
                'max_drawdown': float(val_metrics['max_drawdown']),
                'action_dist': val_metrics['action_dist'],
                'steps':val_metrics['steps']
            }, indent=2)}")
            
            if val_metrics['mean_return'] > agent.manager.best_reward:
                agent.manager.best_reward = val_metrics['mean_return']
                agent.manager.save_checkpoint(episode, is_best=True, suffix='_best')
            
            with open(os.path.join(METRICS_DIR, f'val_metrics_ep{episode}.json'), 'w') as f:
                json.dump(val_metrics, f)
            
            plot_training_metrics(agent, episode, val_metrics)
            agent.training_history['action_dist'] = np.zeros(ACTION_DIM)

        logger.info(f"Episode {episode} | Reward: {total_reward:.2f} | D Loss: {d_loss:.2f} | G Loss: {dqn_loss+adv_loss:.2f}")

def test(model_path):
    """测试模型"""
    test_env = StockTradingEnv(load_and_process_data())
    agent = StockTradingAgent()
    agent.manager.env = test_env
    agent.manager.load_checkpoint(model_path, resume_training=False)
    
    test_metrics = agent.manager.evaluate(n_episodes=5)
    print("\n=== 测试结果 ===")
    print(f"平均回报: {test_metrics['mean_return']:.2f}")
    print(f"动作分布: {dict(zip(['Buy', 'Sell', 'Hold'], test_metrics['action_dist']))}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test('saved_models/model_best.pth')
    else:
        train()



