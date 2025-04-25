from GCA_base import GCABase
import time
import torch
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# from utils.multiGAN_trainer import train_multi_gan
from utils.multiGAN_trainer_disccls import train_multi_gan
from typing import List, Optional
import models
import os
import time
import glob
from utils.evaluate_visualization import evaluate_best_models
import torch.nn.functional as F


def log_execution_time(func):
    """装饰器：记录函数的运行时间，并动态获取函数名"""

    @wraps(func)  # 保留原函数的元信息（如 __name__）
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 动态获取函数名（支持类方法和普通函数）
        func_name = func.__name__
        print(f"GCA_time_series - '{func_name}' elapse time: {elapsed_time:.4f} sec")
        return result

    return wrapper


def generate_labels(y):
    """
    根据每个时间步 y 是否比前一时刻更高，生成三分类标签：
      - 2: 当前值 > 前一时刻（上升）
      - 0: 当前值 < 前一时刻（下降）
      - 1: 当前值 == 前一时刻（平稳）
    对于第一个时间步，默认赋值为1（平稳）。

    参数：
        y: 数组，形状为 (样本数, ) 或 (样本数, 1)
    返回：
        labels: 生成的标签数组，长度与 y 相同
    """
    y = np.array(y).flatten()  # 转成一维数组
    labels = [0]  # 对于第一个样本，默认平稳
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            labels.append(2)
        elif y[i] < y[i - 1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


class GCA_time_series(GCABase):
    def __init__(self, args, N_pairs: int, batch_size: int, num_epochs: int,
                 generators_names: List, discriminators_names: Optional[List],
                 ckpt_dir: str, output_dir: str,
                 window_sizes: int,
                 initial_learning_rate: float = 2e-5,
                 train_split: float = 0.8,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 precise=torch.float32,
                 device=None,
                 seed: int = None,
                 ckpt_path: str = None,
                 gan_weights=None,
                 ):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generators_names: list object，包括了表示具有不同特征的生成器的名称
        :param discriminators_names: list object，包括了表示具有不同判别器的名称，如果没有就不写默认一致
        :param ckpt_dir: 各模型检查点保存目录
        :param output_path: 可视化、损失函数的log等输出目录
        :param ckpt_path: 预测时保存的检查点
        """
        super().__init__(N_pairs, batch_size, num_epochs,
                         generators_names, discriminators_names,
                         ckpt_dir, output_dir,
                         initial_learning_rate,
                         train_split,
                         precise,
                         do_distill_epochs, cross_finetune_epochs,
                         device,
                         seed,
                         ckpt_path)  # 调用父类初始化

        self.args = args
        self.window_sizes = window_sizes
        # 初始化空字典
        self.generator_dict = {}
        self.discriminator_dict = {"default": models.Discriminator3}

        # 遍历 model 模块下的所有属性
        for name in dir(models):
            obj = getattr(models, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                lname = name.lower()
                if "generator" in lname:
                    key = lname.replace("generator_", "")
                    self.generator_dict[key] = obj
                elif "discriminator" in lname:
                    key = lname.replace("discriminator", "")
                    self.discriminator_dict[key] = obj

        self.gan_weights = gan_weights

        self.init_hyperparameters()

    @log_execution_time
    def process_data(self, data_path, start_row, end_row,  target_columns, feature_columns_list):
        """
        Process the input data by loading, splitting, and normalizing it.

        Args:
            data_path (str): Path to the CSV data file
            target_columns (list): Indices of target columns
            feature_columns (list): Indices of feature columns

        Returns:
            tuple: (train_x, test_x, train_y, test_y, y_scaler)
        """
        print(f"Processing data with seed: {self.seed}")  # Using self.seed

        # Load data
        data = pd.read_csv(data_path)

        # Select target columns
        y = data.iloc[start_row:end_row, target_columns].values
        target_column_names = data.columns[target_columns]
        print("Target columns:", target_column_names)

        # # Select feature columns
        # x = data.iloc[start_row:end_row, feature_columns].values
        # feature_column_names = data.columns[feature_columns]
        # print("Feature columns:", feature_column_names)

        # Process each set of feature columns
        x_list = []
        feature_column_names_list = []
        self.x_scalers = []  # Store multiple x scalers

        for feature_columns in feature_columns_list:
            # Select feature columns
            x = data.iloc[start_row:end_row, feature_columns].values
            feature_column_names = data.columns[feature_columns]
            print("Feature columns:", feature_column_names)

            x_list.append(x)
            feature_column_names_list.append(feature_column_names)

        # Data splitting using self.train_split
        train_size = int(data.iloc[start_row:end_row].shape[0] * self.train_split)
        # train_x, test_x = x[:train_size], x[train_size:]
        # Split each x in the list
        train_x_list = [x[:train_size] for x in x_list]
        test_x_list = [x[train_size:] for x in x_list]
        train_y, test_y = y[:train_size], y[train_size:]

        # Normalize each x set separately
        self.train_x_list = []
        self.test_x_list = []
        for train_x, test_x in zip(train_x_list, test_x_list):
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            self.train_x_list.append(x_scaler.fit_transform(train_x))
            self.test_x_list.append(x_scaler.transform(test_x))
            self.x_scalers.append(x_scaler)  # Store all x scalers

        # Normalization
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable

        # self.train_x = self.x_scaler.fit_transform(train_x)
        # self.test_x = self.x_scaler.transform(test_x)

        self.train_y = self.y_scaler.fit_transform(train_y)
        self.test_y = self.y_scaler.transform(test_y)

        # 生成训练集的分类标签（直接在 GPU 上生成）
        self.train_labels = generate_labels(self.train_y)
        # 生成测试集的分类标签
        self.test_labels = generate_labels(self.test_y)
        print(self.train_y[:5])
        print(self.train_labels[:5])
        # ------------------------------------------------------------------

    def create_sequences_combine(self, x_list, y, label, window_size, start):
        x_ = []
        y_ = []
        y_gan = []
        label_gan = []
        # Create sequences for each x in x_list
        for x in x_list:
            x_seq = []
            for i in range(start, x.shape[0]):
                tmp_x = x[i - window_size: i, :]
                x_seq.append(tmp_x)
            x_.append(np.array(x_seq))

        # Combine x sequences along feature dimension
        x_ = np.concatenate(x_, axis=-1)

        for i in range(start, y.shape[0]):
            # tmp_x = x[i - window_size: i, :]
            tmp_y = y[i]
            tmp_y_gan = y[i - window_size: i + 1]
            tmp_label_gan = label[i - window_size: i + 1]

            # x_.append(tmp_x)
            y_.append(tmp_y)
            y_gan.append(tmp_y_gan)
            label_gan.append(tmp_label_gan)

        x_ = torch.from_numpy(np.array(x_)).float()
        y_ = torch.from_numpy(np.array(y_)).float()
        y_gan = torch.from_numpy(np.array(y_gan)).float()
        label_gan = torch.from_numpy(np.array(label_gan)).float()
        return x_, y_, y_gan, label_gan

    @log_execution_time
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""

        # Sliding Window Processing
        # 分别生成不同 window_size 的序列数据
        train_data_list = [
            self.create_sequences_combine(self.train_x_list, self.train_y, self.train_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        test_data_list = [
            self.create_sequences_combine(self.test_x_list, self.test_y, self.test_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        # 分别提取 x、y、y_gan 并堆叠
        self.train_x_all = [x.to(self.device) for x, _, _, _ in train_data_list]
        self.train_y_all = train_data_list[0][1]  # 所有 y 应该相同，取第一个即可，不用cuda因为要eval
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in train_data_list]
        self.train_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in train_data_list]

        self.test_x_all = [x.to(self.device) for x, _, _, _ in test_data_list]
        self.test_y_all = test_data_list[0][1]  # 所有 y 应该相同，取第一个即可，不用cuda因为要eval
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in test_data_list]
        self.test_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in test_data_list]

        assert all(torch.equal(train_data_list[0][1], y) for _, y, _, _ in train_data_list), "Train y mismatch!"
        assert all(torch.equal(test_data_list[0][1], y) for _, y, _, _ in test_data_list), "Test y mismatch!"

        """
        train_x_all.shape  # (N, N, W, F)  不同 window_size 会导致 W 不一样，只能在 W 相同时用 stack
        train_y_all.shape  # (N,)
        train_y_gan_all.shape  # (3, N, W+1)
        """

        self.dataloaders = []

        for i, (x, y_gan, label_gan) in enumerate(
                zip(self.train_x_all, self.train_y_gan_all, self.train_label_gan_all)):
            shuffle_flag = ("transformer" in self.generator_names[i])  # 最后一个设置为 shuffle=True，其余为 False
            dataloader = DataLoader(
                TensorDataset(x, y_gan, label_gan),
                batch_size=self.batch_size,
                shuffle=shuffle_flag,
                generator=torch.manual_seed(self.seed),
                drop_last=True  # 丢弃最后一个不足 batch size 的数据
            )
            self.dataloaders.append(dataloader)

    def init_model(self,num_cls):
        """模型结构初始化"""
        assert len(self.generator_names) == self.N, "Generators and Discriminators mismatch!"
        assert isinstance(self.generator_names, list)
        for i in range(self.N):
            assert isinstance(self.generator_names[i], str)

        self.generators = []
        self.discriminators = []

        for i, name in enumerate(self.generator_names):
            # 获取对应的 x, y
            x = self.train_x_all[i]
            y = self.train_y_all[i]

            # 初始化生成器
            GenClass = self.generator_dict[name]
            if "transformer" in name:
                gen_model = GenClass(x.shape[-1], output_len=y.shape[-1]).to(self.device)
            else:
                gen_model = GenClass(x.shape[-1], y.shape[-1]).to(self.device)

            self.generators.append(gen_model)

            # 初始化判别器（默认只用 Discriminator3）
            DisClass = self.discriminator_dict[
                "default" if self.discriminators_names is None else self.discriminators_names[i]]
            dis_model = DisClass(self.window_sizes[i], out_size=y.shape[-1], num_cls=num_cls).to(self.device)
            self.discriminators.append(dis_model)

    def init_hyperparameters(self, ):
        """初始化训练所需的超参数"""
        # 初始化：对角线上为1，其余为0，最后一列为1.0
        self.init_GDweight = []
        for i in range(self.N):
            row = [0.0] * self.N
            row[i] = 1.0
            row.append(1.0)  # 最后一列为 scale
            self.init_GDweight.append(row)

        if self.gan_weights is None:
            # 最终：均分组合，最后一列为1.0
            final_row = [round(1.0 / self.N, 3)] * self.N + [1.0]
            self.final_GDweight = [final_row[:] for _ in range(self.N)]
        else:
            pass

        self.g_learning_rate = self.initial_learning_rate
        self.d_learning_rate = self.initial_learning_rate
        self.adam_beta1, self.adam_beta2 = (0.9, 0.999)
        self.schedular_factor = 0.1
        self.schedular_patience = 16
        self.schedular_min_lr = 1e-7

    def train(self, logger):
        results, best_model_state = train_multi_gan(self.args, self.generators, self.discriminators, self.dataloaders,
                                                    self.window_sizes,
                                                    self.y_scaler, self.train_x_all, self.train_y_all, self.test_x_all,
                                                    self.test_y_all,
                                                    self.do_distill_epochs,self.cross_finetune_epochs,
                                                    self.num_epochs,
                                                    self.output_dir,
                                                    self.device,
                                                    init_GDweight=self.init_GDweight,
                                                    final_GDweight=self.final_GDweight,
                                                    logger=logger)

        self.save_models(best_model_state)
        return results

    def save_models(self, best_model_state):
        """
        保存所有 generator 和 discriminator 的模型参数，包含时间戳、模型名称或编号。
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, timestamp)
        gen_dir = os.path.join(ckpt_dir, "generators")
        disc_dir = os.path.join(ckpt_dir, "discriminators")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(disc_dir, exist_ok=True)

        # 加载模型并设为 eval
        for i in range(self.N):
            self.generators[i].load_state_dict(best_model_state[i])
            self.generators[i].eval()

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt")
            torch.save(gen.state_dict(), save_path)

        for i, disc in enumerate(self.discriminators):
            disc_name = type(disc).__name__
            save_path = os.path.join(disc_dir, f"{i + 1}_{disc_name}.pt")
            torch.save(disc.state_dict(), save_path)

        print("All models saved with timestamp and identifier.")

    def get_latest_ckpt_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        all_subdirs = [d for d in glob.glob(os.path.join(self.ckpt_dir, timestamp[0] + "*")) if os.path.isdir(d)]
        if not all_subdirs:
            raise FileNotFoundError("❌ No checkpoint records!!")
        latest = max(all_subdirs, key=os.path.getmtime)
        print(f"📂 Auto loaded checkpoint file: {latest}")
        return latest

    def load_model(self):
        gen_path = os.path.join(self.ckpt_path, "g{gru}", "generator.pt")
        if os.path.exists(gen_path):
            self.generators[0].load_state_dict(torch.load(gen_path, map_location=self.device))
            print(f"✅ Loaded generator from {gen_path}")
        else:
            raise FileNotFoundError(f"❌ Generator checkpoint not found at: {gen_path}")

    def pred(self):
        if self.ckpt_path == "auto":
            self.ckpt_path = self.get_latest_ckpt_folder()

        print("Start predicting with all generators..")
        best_model_state = [None for _ in range(self.N)]
        current_path = os.path.join(self.ckpt_path, "generatorrs")

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(current_path, f"{i + 1}_{gen_name}.pt")
            best_model_state[i] = self.generator_dict[self.generator_names[i]].load_state_dict(torch.load(save_path))

        results = evaluate_best_models(self.generators, best_model_state, self.train_x_all, self.train_y_all,
                                       self.test_x_all, self.test_y_all, self.y_scaler,
                                       self.output_dir)
        return results

    def distill(self):
        """评估模型性能并可视化结果"""
        pass

    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass
