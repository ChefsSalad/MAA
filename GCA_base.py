from abc import ABC, abstractmethod
import random, torch, numpy as np
from utils.util import setup_device
import os

class GCABase(ABC):
    """
    GCA 框架的虚基类，定义核心方法接口。
    所有子类必须实现以下方法。
    """

    def __init__(self, N_pairs, batch_size, num_epochs,
                 generator_names, discriminators_names,
                 ckpt_dir, output_dir,
                 initial_learning_rate = 2e-4,
                 train_split = 0.8,
                 precise = torch.float32,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 device = None,
                 seed=None,
                 ckpt_path="auto",):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generators: 建议是一个iterable object，包括了表示具有不同特征的生成器
        :param discriminators: 建议是一个iterable object，可以是相同的判别器
        :param ckpt_path: 各模型检查点
        :param output_path: 可视化、损失函数的log等输出路径
        """

        self.N = N_pairs
        self.initial_learning_rate = initial_learning_rate
        self.generator_names = generator_names
        self.discriminators_names = discriminators_names
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_split = train_split
        self.seed = seed
        self.do_distill_epochs = do_distill_epochs
        self.cross_finetune_epochs = cross_finetune_epochs
        self.device = device
        self.precise = precise

        self.set_seed(self.seed)  # 初始化随机种子
        self.device = setup_device(device)
        print("Running Device:", self.device)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("Output directory created! ")

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            print("Checkpoint directory created! ")

    def set_seed(self, seed):
        """
        设置随机种子以确保实验的可重复性。

        :param seed: 随机种子
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def process_data(self):
        """数据预处理，包括读取、清洗、划分等"""
        pass

    @abstractmethod
    def init_model(self):
        """模型结构初始化"""
        pass

    @abstractmethod
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""
        pass

    @abstractmethod
    def init_hyperparameters(self):
        """初始化训练所需的超参数"""
        pass

    @abstractmethod
    def train(self):
        """执行训练过程"""
        pass

    @abstractmethod
    def save_models(self):
        """执行训练过程"""
        pass

    @abstractmethod
    def distill(self):
        """执行知识蒸馏过程"""
        pass

    @abstractmethod
    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    @abstractmethod
    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass
