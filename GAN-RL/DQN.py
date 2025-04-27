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
# ==================== 设备配置 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
# 超参数配置
SEQ_LENGTH = 10
PRICE_FEATURES = 4
INPUT_DIM = 58  # 40(raw) +12(patterns)+5(memory)+1(position)
ACTION_DIM = 3
HIDDEN_DIM = 128
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.9
MEMORY_SIZE = 10000
TAU = 0.01  # 软更新参数

# 路径配置
MODEL_DIR = "dqn_models"
METRICS_DIR = "dqn_metrics"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ==================== 模型组件 ====================
class D3QN(nn.Module):
    """Dueling Double DQN网络"""
    def __init__(self, input_dim = INPUT_DIM, action_dim=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM*2),
            nn.LayerNorm(HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2)
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
        V = self.value_stream(fc_out)
        A = self.advantage_stream(fc_out)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q, V, A

# ==================== 智能体与训练逻辑 ====================
class DQNAgent:
    def __init__(self):
        self.online_net = D3QN(INPUT_DIM, ACTION_DIM).to(device)
        self.target_net = D3QN(INPUT_DIM, ACTION_DIM).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.manager = ModelManager(self, None)
        self._init_training_history()

        self.epsilon_start = 1.0   # 初始探索率
        self.epsilon_end = 0.08    # 最终探索率
        self.epsilon_decay = 200    # 衰减周期（指数形式）
        self.current_episode = 0   # 新增episode计数器

    def _init_training_history(self):
        self.training_history = {
            'rewards': [],
            'base_rewards': [],
            'pattern_rewards': [],  
            'losses': [],
            'grad_norms': [],
            'action_dist': np.zeros(ACTION_DIM)
        }

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # states = torch.FloatTensor(np.array(states))
        # next_states = torch.FloatTensor(np.array(next_states))
        # actions = torch.LongTensor(actions)
        # rewards = torch.FloatTensor(rewards)
        # dones = torch.FloatTensor(dones)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        # 计算当前Q值
        current_q, _, _ = self.online_net(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))

        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            next_q, _, _ = self.online_net(next_states)
            best_actions = next_q.argmax(1)
            target_q, _, _ = self.target_net(next_states)
            target_q = target_q.gather(1, best_actions.unsqueeze(1))
            target = rewards + (1 - dones) * GAMMA * target_q.squeeze()

        # 计算损失
        loss = F.mse_loss(current_q.squeeze(), target)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 2.0)
        self.optimizer.step()

        # 记录梯度
        grad_norms = [p.grad.norm().item() for p in self.online_net.parameters() if p.grad is not None]
        self.training_history['grad_norms'].append(np.mean(grad_norms) if grad_norms else 0)

        # 软更新目标网络
        self.soft_update_targets()
        
        return loss.item()

    def soft_update_targets(self):
        for t_param, o_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_param.data.copy_(TAU * o_param.data + (1 - TAU) * t_param.data)

    def get_action(self, state, training = True):
        if training:
        # 指数衰减公式
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                    np.exp(-self.current_episode / self.epsilon_decay)
        else:
            epsilon = 0.001  # 测试时使用固定小概率探索
        
        if np.random.rand() < epsilon:
            return np.random.randint(ACTION_DIM)
        
        with torch.no_grad():
            # state_tensor = torch.FloatTensor(state).unsqueeze(0)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values, _, _ = self.online_net(state_tensor)
            action = q_values.argmax().item()
            self.training_history['action_dist'][action] += 1
            return action

class ModelManager:
    """模型管理组件"""
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.best_reward = -np.inf
        self.best_metrics = {}
        self.history = {'train': [], 'val': [], 'test': []}

    def save_checkpoint(self, episode, is_best=False):
        checkpoint = {
            'episode': episode,
            'online_net': self.agent.online_net.state_dict(),
            'target_net': self.agent.target_net.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'training_history': self.agent.training_history,
            'best_reward': self.best_reward
        }
        filename = f'dqn_checkpoint_ep{episode}.pth'
        torch.save(checkpoint, os.path.join(MODEL_DIR, filename))
        if is_best:
            torch.save(checkpoint, os.path.join(MODEL_DIR, 'dqn_best.pth'))

    def load_checkpoint(self, path):
        # checkpoint = torch.load(path)
        checkpoint = torch.load(path, map_location=device)
        self.agent.online_net.load_state_dict(checkpoint['online_net'])
        self.agent.target_net.load_state_dict(checkpoint['target_net'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.agent.training_history = checkpoint['training_history']
        self.best_reward = checkpoint['best_reward']
        logger.info(f"加载检查点，最佳奖励：{self.best_reward:.2f}")
    def plot_trading_decision(self, prices, actions, holdings, episode):
        """绘制交易决策图表"""
        plt.figure(figsize=(15, 8))
        
        # 创建价格和动作子图
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(prices, label='期货价格', color='#1f77b4', linewidth=1.5)
        
        # 标记交易动作
        buy_points = [i for i, a in enumerate(actions) if a == 0]
        sell_points = [i for i, a in enumerate(actions) if a == 1]
        close_points = [i for i, a in enumerate(actions) if a == 2]
        
        ax1.scatter(buy_points, [prices[i] for i in buy_points], 
                color='green', marker='^', s=80, label='买入', alpha=0.7)
        ax1.scatter(sell_points, [prices[i] for i in sell_points], 
                color='red', marker='v', s=80, label='卖出', alpha=0.7)
        ax1.scatter(close_points, [prices[i] for i in close_points], 
                color='purple', marker='o', s=60, label='平仓', alpha=0.7)
        
        ax1.set_title(f'Episode {episode} 交易决策可视化', fontsize=14)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.legend()
        
        # 创建持仓子图
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.bar(range(len(holdings)), holdings, 
            color=np.where(np.array(holdings)>=0, 'green', 'red'), 
            alpha=0.6, width=1.0)
        ax2.set_title('持仓变化', fontsize=14)
        ax2.set_xlabel('时间步', fontsize=12)
        ax2.set_ylabel('持仓量', fontsize=12)
        ax2.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_DIR, f'trading_decision_ep{episode}.png'))
        plt.close()
        
    def evaluate(self, n_episodes=10):
        metrics = {
            'total_returns': [],
            'action_dist': np.zeros(ACTION_DIM),
            'steps': [],
            'trading_details': []  # 新增交易细节记录
        }

        # 创建评估结果目录
        eval_dir = os.path.join(METRICS_DIR, "evaluation_details")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 保存当前episode的详细信息
        filename = f"eval_episode.txt"
        filepath = os.path.join(eval_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump("", f, indent=2)
        except Exception as e:
            logger.error(f"更新评估详情失败: {str(e)}")
        
        
        for ep in range(n_episodes):
            state = self.env.reset()
            done = False
            portfolio_value = [self.env.init_balance]

            # 新增数据记录
            episode_prices = []
            episode_actions = []
            episode_holdings = []

            while not done:
                # 记录步骤开始时的状态
                step_info = {
                    # "step": self.env.current_step,
                    # "position_before": {
                    #     "balance": float(self.env.balance),
                    #     "holdings": float(self.env.holdings),
                    #     "value": float(self.env.balance + self.env.holdings * 
                    #                  self.env.combined_data[self.env.current_step, 3])
                    # }
                }
                action = self.agent.get_action(state, False)
                next_state, reward, done, _ = self.env.step(action)
                current_value = self.env.balance + self.env.holdings * self.env.combined_data[self.env.current_step-1, 3]
                portfolio_value.append(current_value)
                metrics['action_dist'][action] += 1

                current_price = self.env.combined_data[self.env.current_step-1, 3]
                episode_prices.append(current_price)
                episode_actions.append(action)
                episode_holdings.append(self.env.holdings)

                state = next_state
                 # 记录步骤执行后的状态
                step_info.update({
                    "step": self.env.current_step,
                    "action": int(action),
                    "reward": float(reward),
                    "debt": self.env.debt,
                    "position_after": {
                        "balance": float(self.env.balance),
                        "holdings": float(self.env.holdings),
                        "value": (self.env.balance - self.env.debt + self.env.holdings * 
                                     self.env.combined_data[self.env.current_step, 3])
                    },
                },)


                
                try:
                    with open(filepath, 'a') as f:
                        json.dump(step_info, f, indent=2)
                except Exception as e:
                    logger.error(f"保存评估详情失败: {str(e)}")

             # 保存交易细节
            metrics['trading_details'].append({
                'prices': episode_prices,
                'actions': episode_actions,
                'holdings': episode_holdings
            })

            returns = (portfolio_value[-1] - self.env.init_balance) / self.env.init_balance
            metrics['total_returns'].append(returns)
            metrics['steps'].append(portfolio_value)

        
    
        return {
            'mean_return': np.mean(metrics['total_returns']),
            'std_return': np.std(metrics['total_returns']),
            'action_dist': metrics['action_dist'].tolist(),
            'steps': metrics['steps'],
            'trading_details': metrics['trading_details']  # 返回交易细节
        }

# ==================== 数据与环境 ====================
def load_and_process_data():

    price_df = pd.read_csv('Soybean.csv', parse_dates=['time'])
    price_df = price_df[['time', 'open', 'high', 'low', 'close']].set_index('time')
    price_df = price_df[['open', 'high', 'low', 'close']].values.astype(np.float32)
        
    return price_df

# ==================== 增强型K线特征模块 ====================
class EnhancedCandleFeatures:
    pattern_threshold = 0.5  # 形态检测阈值
    
    @staticmethod
    def detect_patterns(window):
        """输入: 最近N根K线数据 (SEQ_LENGTH x 4)"""
        features = np.zeros(12, dtype=np.float32)
        
        # 获取最近3根K线
        curr = window[-1]
        prev1 = window[-2] if len(window)>=2 else None
        prev2 = window[-3] if len(window)>=3 else None

        # 单K线特征
        body = abs(curr[3]-curr[0])
        upper = curr[1]-max(curr[0],curr[3])
        lower = min(curr[0],curr[3])-curr[2]
        
        # 锤子线/上吊线
        features[0] = 1 if (lower>2*body and upper<0.2*body) else 0
        features[1] = 1 if (upper>2*body and lower<0.2*body) else 0

        # 吞没模式（需要前1根）
        if prev1 is not None:
            bull_engulf = (curr[3]>prev1[0] and curr[0]<prev1[3]) and (curr[3]-curr[0])>0.6*(prev1[0]-prev1[3])
            bear_engulf = (curr[3]<prev1[0] and curr[0]>prev1[3]) and (curr[0]-curr[3])>0.6*(prev1[3]-prev1[0])
            features[2] = 1 if bull_engulf else 0
            features[3] = 1 if bear_engulf else 0

        # 可以进一步加入其他特征检测（晨星、三兵等），尚未实现
        
        return features


class StockTradingEnv:
    """股票交易环境"""
    def __init__(self, combined_data, seq_length=10):
        self.combined_data = combined_data
        # self.price_data = price_data
        self.seq_length = seq_length
        self.current_step = seq_length
        self.position = 0.0
        self.debt = 0 # 借入债务
        self.leverage = 3  # 杠杆倍数
        self.max_risk_per_trade = 0.02  # 单笔交易最大风险（账户价值的2%）
        self.stop_loss_pct = 0.01      # 止损点位（价格反向波动的1%）

        self.transFee = 0 # 添加交易手续费
        self.init_balance = 500000  # 初始资金

        # 噪声配置参数
        self.noise_level = 0.01  # 初始噪声水平
        self.noise_type = 'gaussian'  # 可选 'uniform'
        self.dynamic_noise = True  # 是否启用动态噪声调整

    def _add_noise(self, state):
        """为输入状态添加可控噪声"""
        price_part = state[:SEQ_LENGTH*PRICE_FEATURES]  
        if self.noise_type == 'gaussian':
            noise = np.random.normal(
                # scale=self.noise_level * np.std(price_part, axis=0),
                scale=self.noise_level,
                size=price_part.shape
            )
        else:
            noise = np.random.uniform(
                low=-self.noise_level,
                high=self.noise_level,
                size=price_part.shape
            )
  
        # 仅对价格部分添加噪声
        noisy_price = price_part + noise
        # noisy_price[:, :3] = np.clip(noisy_price[:, :3], a_min=0, a_max=None)
        noisy_price = np.clip(noisy_price, a_min=0, a_max=None)
        
        # 重新拼接完整状态
        new_state = np.concatenate([
            # noisy_price.flatten(),
            noisy_price,
            state[SEQ_LENGTH*PRICE_FEATURES:]
        ], dtype=np.float32)

        return new_state
        # return new_state.astype(np.float32)
    
    def _update_noise_level(self, episode):
        """动态调整噪声水平（在训练循环中调用）"""
        if self.dynamic_noise:
            # 随训练进程线性衰减噪声
            self.noise_level = max(0.001, 0.05 * (1 - episode/1000)) 
    
    def reset(self):
        self.balance = self.init_balance
        self.holdings = 0
        self.portfolio_values = []
        self.debt = 0
        
        self.current_step = self.seq_length
        self.position = 0
        return self._get_state()

    def step(self, action):
        prev_close = self.combined_data[self.current_step-2, 3]  
        current_close = self.combined_data[self.current_step-1, 3]
   
        if action == 0:  # 买入开多仓
            # if self.balance > self.transFee:
#                 # 风险控制计算
#                 max_allowable_loss = current_value * self.max_risk_per_trade
#                 price_risk = current_close * self.stop_loss_pct
#                 max_position_by_risk = max_allowable_loss / price_risk

#                 # 杠杆计算的最大可买量
#                 max_leverage_position = ((current_value * self.leverage) - self.transFee) / current_close

#                 # 取风险限制和杠杆限制的较小值
#                 actual_position = min(max_position_by_risk, max_leverage_position)
                
#                 self.holdings += actual_position
#                 self.balance = max(current_value - actual_position * current_close - self.transFee, 0)
            current_value = self.balance + self.holdings * current_close
            net_asset = current_value - self.debt  # 计算净资产（扣除已有负债）
            new_debt = max(net_asset * (self.leverage - 1) - self.debt, 0)
            if new_debt + self.balance > 0:
                max_afford = (new_debt + self.balance - self.transFee) / current_close
                self.holdings += max_afford
                self.debt += new_debt
                self.balance = 0
        elif action == 1:  # 卖出开空仓
            if self.holdings > 0:  # 平多仓
                sell_value = self.holdings * current_close
                if sell_value >= self.transFee:
                    net_proceeds = sell_value - self.transFee
                    repay = min(net_proceeds, self.debt)
                    self.debt -= repay
                    self.balance += (net_proceeds - repay)
                self.holdings = 0
            if self.balance >= self.transFee:
                max_afford = (self.balance - self.transFee) / current_close
                self.holdings -= max_afford  # 持仓变为负数表示空头
                self.balance += max_afford * current_close - self.transFee  # 卖出获得资金
        elif action == 2:  # 平仓
            if self.holdings > 0:  # 平多仓
                sell_value = self.holdings * current_close
                if sell_value >= self.transFee:
                    net_proceeds = sell_value - self.transFee
                    repay = min(net_proceeds, self.debt)
                    self.debt -= repay
                    self.balance += (net_proceeds - repay)
                self.holdings = 0
            elif self.holdings < 0:  # 平空仓
                required_cash = abs(self.holdings) * current_close + self.transFee
                if self.balance >= required_cash:
                    self.balance -= required_cash
                    self.holdings = 0

        
        # 添加趋势跟随奖励
        trend_reward = 0
        ma_short = np.mean(self.combined_data[self.current_step-5:self.current_step, 3])
        ma_long = np.mean(self.combined_data[self.current_step-20:self.current_step, 3])
        if (action == 0) and (ma_short > ma_long):  # 趋势向上时买入奖励
            trend_reward += 0.5
        elif (action == 1) and (ma_short < ma_long):  # 趋势向下时卖出奖励
            trend_reward += 0.5

        # 计算新奖励函数
        current_value = self.balance + self.holdings * current_close - self.debt
        self.portfolio_values.append(current_value)

        reward = self._calculate_reward(current_value) + trend_reward

        self.current_step += 1  # 移动到下一个时间步
        done = self.current_step >= len(self.combined_data) - 1
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, current_value):
        # 1. 对数收益率 (更稳定)
        if len(self.portfolio_values) >= 2:
            returns = (current_value - self.init_balance) / 50000
        else:
            returns = 0.0

        # # 2. 动态夏普率奖励 (滑动窗口)
        # sharpe_reward = 0
        # window_size = 30
        # if len(self.portfolio_values) > window_size:
        #     returns_series = np.diff(np.log(self.portfolio_values[-window_size:])) * 100
        #     excess_returns = returns_series - 0.02/window_size  # 假设无风险利率2%年化
        #     sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)
        #     sharpe_reward = sharpe * 0.3  # 调整权重

        # # 3. 回撤惩罚 (动态计算)
        # peak = np.max(self.portfolio_values)
        # drawdown = (peak - current_value) / peak if peak > 0 else 0
        # drawdown_penalty = -drawdown * 0.8  # 加大惩罚力度

        # # 4. 波动率惩罚
        # volatility_penalty = -np.std(np.diff(np.log(self.portfolio_values[-30:])))*100 if len(self.portfolio_values)>30 else 0

        # # 5. 持仓变化惩罚 (减少频繁交易)
        # position_change = abs(self.holdings - self.prev_holdings) if hasattr(self, 'prev_holdings') else 0
        # turnover_penalty = -position_change * 0.1
        # self.prev_holdings = self.holdings

        # # 6. 胜率奖励 (长期窗口)
        # win_rate_reward = 0
        # if len(self.portfolio_values) > 100:
        #     positive_returns = np.sum(np.diff(self.portfolio_values[-100:]) > 0)
        #     win_rate = positive_returns / 99
        #     win_rate_reward = 2.0 * (win_rate - 0.5)  # 胜率超过50%才奖励

        # 组合奖励
        total_reward = returns * 0.5
            # sharpe_reward + 
            # drawdown_penalty + 
            # volatility_penalty * 0.3 + 
            # turnover_penalty + 
            # win_rate_reward
        return total_reward

    def _get_state(self):
        raw_state = self.combined_data[self.current_step-self.seq_length : self.current_step].flatten()
        return self._add_noise(raw_state)

# ==================== K线驱动环境 ==================== 
class PureCandleEnv(StockTradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_memory = deque([0.0]*5, maxlen=5)  # 初始填充5个0

    def _get_state(self):
        raw = self.combined_data[self.current_step-self.seq_length : self.current_step]
        patterns = EnhancedCandleFeatures.detect_patterns(raw)
        
        # 确保memory部分始终为5维
        memory = list(self.pattern_memory)[-5:]  # 取最后5个元素
        if len(memory) < 5:  # 填充不足部分
            memory += [0.0]*(5 - len(memory))

        # 构建新状态向量
        state = np.concatenate([
            raw.flatten(),               # 原始价格序列 (10x4=40)
            patterns,                    # K线特征 (12)
            self.pattern_memory,         # 形态记忆 (5)
            [self.holdings/self.init_balance]  # 归一化持仓 (1)
        ], dtype=np.float32)
        
        return self._add_noise(state)
    
    def step(self, action):
        old_step = super().step(action)
        current_pattern = EnhancedCandleFeatures.detect_patterns(
            self.combined_data[self.current_step-3:self.current_step]
        )
        logger.debug(
            "Step %d: 动作=%d, 仓位=%.2f, 余额=%.2f, 价值变化: %.2f → %.2f",
            self.current_step, action, self.holdings, self.balance
        )
        self.pattern_memory.append(np.mean(current_pattern[:8]))
        
        return (*old_step[:3], {'pattern': current_pattern})  
    
    def _pattern_reward(self, action, patterns):
        """K线模式奖励"""
        reward = 0
        
        # 持仓状态映射
        position = self.holdings / self.init_balance  # 归一化仓位
        
        # # 基础模式奖励
        # reward += patterns[2] * 1.5   # 看涨吞没
        # reward += patterns[3] * -1.5  # 看跌吞没
        # reward += patterns[4] * 2.0   # 晨星
        # reward += patterns[5] * -2.0  # 暮星
        
        # 仓位匹配奖励
        if position > 0.1:  # 持多仓
            reward += patterns[0] * 0.5  # 锤子线奖励
            reward -= patterns[1] * 1.0  # 上吊线惩罚
        elif position < -0.1:  # 持空仓
            reward += patterns[1] * 0.5  # 上吊线奖励
            reward -= patterns[0] * 1.0  # 锤子线惩罚
            
        return reward  # 控制奖励规模


# ==================== 训练与测试 ====================
def plot_training_metrics(agent, episode, val_metrics=None):
    plt.figure(figsize=(18, 12))
    sns.set_style("whitegrid")
    
    plt.subplot(2, 2, 1)
    plt.plot(agent.training_history['rewards'])
    plt.title('Episode Rewards')
    
    plt.subplot(2, 2, 2)
    plt.plot(agent.training_history['losses'])
    plt.title('DQN Loss')
    
    plt.subplot(2, 2, 3)
    actions = ['Buy', 'Sell', 'Balance']
    plt.bar(actions, agent.training_history['action_dist'], color=['green', 'red', 'blue'])
    plt.title('Action Distribution')
    
    if val_metrics:
        plt.subplot(2, 2, 4)
        plt.bar(['Return'], [val_metrics['mean_return']], color='green')
        plt.title('Validation Returns')
    
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, f'dqn_metrics_ep{episode}.png'))
    plt.close()

def visualize_pattern_impact(agent, env, n_episodes=5):
    pattern_actions = {i:[] for i in range(12)}
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, False)
            patterns = state[40:52]  # 提取特征部分
            
            for i in range(12):
                if patterns[i] > EnhancedCandleFeatures.pattern_threshold:
                    pattern_actions[i].append(action)
                    
            state, _, done, _ = env.step(action)
    
    # 绘制影响图
    plt.figure(figsize=(15,10))
    patterns = ['Hammer', 'Hanging', 'BullEngulf', 'BearEngulf',
               'MorningStar', 'EveningStar', 'ThreeWhite', 'ThreeBlack',
               'Doji', 'BullDensity', 'BearDensity', 'NetDensity']
    
    for i in range(12):
        plt.subplot(3,4,i+1)
        if pattern_actions[i]:
            counts = np.bincount(pattern_actions[i], minlength=3)
            plt.bar(['Buy','Sell','Hold'], counts, color=['g','r','b'])
        plt.title(f'{patterns[i]} Impact')
    
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, 'pattern_impact.png'))

def train():
    data = load_and_process_data()
    # env = StockTradingEnv(data)  # 使用原始环境类
    env = PureCandleEnv(data)  # 使用新的环境类
    agent = DQNAgent()
    manager = ModelManager(agent, env)

    # 新增课程学习参数
    PHASE_SETTINGS = [
        (0, 300, 0.001, 0.7),   # 阶段1：基础学习
        (300, 700, 0.0005, 0.8),# 阶段2：增强学习
        (700, 1000, 0.0002, 0.9)# 阶段3：微调
    ]

    for episode in range(1000):
        for start, end, lr, threshold in PHASE_SETTINGS:
            if start <= episode < end:
                EnhancedCandleFeatures.pattern_threshold = threshold
                for g in agent.optimizer.param_groups:
                    g['lr'] = lr
                break
        state = env.reset()
        total_base_reward = 0
        total_pattern_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, base_reward, done, info = env.step(action)
            current_pattern = info['pattern']
            pattern_reward = env._pattern_reward(action, current_pattern)
            # 组合奖励
            combined_reward = base_reward * 0.005 + pattern_reward * 20
            agent.memory.append((state, action, combined_reward, next_state, done))
            state = next_state
            total_base_reward += base_reward * 0.005
            total_pattern_reward += pattern_reward * 20

            if len(agent.memory) >= BATCH_SIZE:
                loss = agent.update()
                if len(agent.memory) % 10 == 0:
                    logger.info(f"Loss: {loss:.5f} | Grad Norm: {np.mean(agent.training_history['grad_norms'][-10:])}")

        # 记录训练指标
        agent.training_history['rewards'].append(total_base_reward + total_pattern_reward)
        agent.training_history['losses'].append(loss if 'loss' in locals() else 0)
        agent.training_history['base_rewards'].append(total_base_reward)
        agent.training_history['pattern_rewards'].append(total_pattern_reward)
        # 定期评估和保存
        if episode % 5 == 0:
            val_metrics = manager.evaluate(n_episodes=5)
            logger.info(f"{json.dumps({'mean_return': float(val_metrics['mean_return']),'std_return': float(val_metrics['std_return']),'action_dist': val_metrics['action_dist']}, indent=2)}")
            
#             logger.info(f"验证回报：{val_metrics['mean_return']:.2f}±{val_metrics['std_return']:.2f}")
            
            if val_metrics['mean_return'] > manager.best_reward:
                manager.best_reward = val_metrics['mean_return']
                manager.save_checkpoint(episode, is_best=True)
            
            plot_training_metrics(agent, episode, val_metrics)
            agent.training_history['action_dist'] = np.zeros(ACTION_DIM)

        logger.info(f"Episode {episode} | (Base: {total_base_reward:.1f}, Pattern: {total_pattern_reward:.1f})")
def test(model_path):
    """测试模型"""
    data = load_and_process_data()
    test_env = PureCandleEnv(data)  # 使用增强后的环境
    agent = DQNAgent()
    agent = DQNAgent()
    agent.manager.env = test_env
    agent.manager.load_checkpoint(model_path)
    
    test_metrics = agent.manager.evaluate(n_episodes=5)
    for ep, detail in enumerate(test_metrics['trading_details']):
        agent.manager.plot_trading_decision(
            detail['prices'],
            detail['actions'],
            detail['holdings'],
            ep+1
        )

    print("\n=== 测试结果 ===")
    print(f"平均回报: {test_metrics['mean_return']:.2f}")
    print(f"动作分布: {dict(zip(['Buy', 'Sell', 'Hold'], test_metrics['action_dist']))}")
    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test('dqn_models/dqn_best.pth')
    else:
        train()