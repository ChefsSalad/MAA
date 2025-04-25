import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Generator_gru(nn.Module):
    def __init__(self, input_size, out_size, hidden_dim = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)  # 仅保留一层GRU，隐藏单元数为256
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear_2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.linear_3 = nn.Linear(hidden_dim//4, out_size)
        self.dropout = nn.Dropout(0.2)

        # 添加分类头，输入维度为256，输出3类别
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim//2, 3)
        )

    def forward(self, x):
        device = x.device
        # 初始化GRU隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_dim, device=device)
        # 通过GRU层
        out, _ = self.gru(x, h0)
        # 取序列最后一个时间步的输出，并经过dropout处理
        last_feature = self.dropout(out[:, -1, :])

        # 原始输出（例如生成或回归任务）
        gen = self.linear_1(last_feature)
        gen = self.linear_2(gen)
        gen = self.linear_3(gen)

        # 分类输出
        cls = self.classifier(last_feature)

        return gen, cls


# class Generator_lstm(nn.Module):
#     def __init__(self, input_size, out_size):
#         super().__init__()
#         self.cnn = nn.Conv1d(in_channels=input_size, out_channels=input_size * 8, kernel_size=3, padding=1)
#         self.lstm = nn.LSTM(input_size=input_size * 8, hidden_size=128,
#                             num_layers=2, batch_first=True, dropout=0.2)
#         self.linear = nn.Linear(128, out_size)
#
#         # 分类头：输入128维，输出3类别
#         self.classifier = nn.Linear(128, 3)
#
#     def forward(self, x, hidden=None):
#         # 调整维度以适应Conv1d：(batch, features, seq_len)
#         x = x.permute(0, 2, 1)
#         cnn_out = nn.LeakyReLU()(self.cnn(x))
#         # 恢复序列格式 (batch, seq_len, features)
#         cnn_out = cnn_out.permute(0, 2, 1)
#         lstm_out, hidden = self.lstm(cnn_out, hidden)
#         # 自适应池化得到固定长度输出
#         pooled_out = F.adaptive_avg_pool1d(lstm_out.permute(0, 2, 1), 1).squeeze(2)
#
#         # 原始输出
#         gen = self.linear(pooled_out)
#         # 分类输出
#         cls = self.classifier(pooled_out)
#
#         return gen, cls


class Generator_lstm(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=128, num_layers=1, dropout=0.1):
        """
        Args:
            input_size (int): 输入特征数
            out_size (int): 输出目标维度（例如用于生成回归结果）
            hidden_size (int): LSTM 的隐藏单元数
            num_layers (int): LSTM 层数，默认 1 层（减少计算量）
            dropout (float): LSTM 内部 dropout 系数，默认为 0.1
        """
        super().__init__()
        # 使用深度可分离卷积：先进行 depthwise, 再 pointwise 转换
        self.depth_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size,
                                    kernel_size=3, padding=1, groups=input_size)
        self.point_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.act = nn.ReLU()

        # LSTM 部分：输入通道数为 (input_size * 4)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        # 直接使用最后一个时间步的输出进行线性映射
        self.linear = nn.Linear(hidden_size, out_size)
        # # 添加分类头，输入维度为256，输出3类别
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Linear(hidden_size//2, 3)
        # )

        self.classifier = nn.Linear(hidden_size, 3)


    def forward(self, x, hidden=None):
        """
        Args:
            x (torch.Tensor): 输入，形状 (batch_size, seq_len, input_size)
            hidden: 可选的 LSTM 初始状态
        Returns:
            torch.Tensor: 输出，形状 (batch_size, out_size)
        """
        # 调整维度：将输入从 (B, T, F) 转为 (B, F, T) 以适应Conv1d
        x = x.permute(0, 2, 1)  # (B, input_size, T)
        # 深度卷积
        x = self.depth_conv(x)
        # 点卷积
        x = self.point_conv(x)
        x = self.act(x)
        # 转回 (B, T, F')
        x = x.permute(0, 2, 1)
        # LSTM 前向传播：这里使用默认最后一时刻状态作为输出
        lstm_out, hidden = self.lstm(x, hidden)
        # 直接取最后一个时间步输出作为特征（避免额外池化操作）
        last_out = lstm_out[:, -1, :]
        out = self.linear(last_out)
        cls = self.classifier(last_out)

        return out, cls

# 位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        """
        model_dim: 模型的特征向量维度
        max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, model_dim)

        # 位置索引
        positions = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]

        # 维度索引，使用指数函数缩放
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))

        # 偶数位置使用 sin，奇数位置使用 cos
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # 增加 batch 维度：[1, max_len, model_dim]

    def forward(self, x):
        """
        x: 输入特征 [batch_size, seq_len, model_dim]
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)  # 只取对应长度的位置信息


class Generator_transformer(nn.Module):
    def __init__(self, input_dim, feature_size=128, num_layers=2, num_heads=8, dropout=0.1, output_len=1):
        """
        input_dim: 数据特征维度
        feature_size: 模型特征维度
        num_layers: 编码器层数
        num_heads: 注意力头数目
        dropout: dropout概率
        output_len: 预测时间步长度（原始任务输出维度）
        """
        super().__init__()
        self.feature_size = feature_size
        self.output_len = output_len
        self.input_projection = nn.Linear(input_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        # 添加 batch_first=True
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_len)  # 原始任务输出
        # 添加分类头：输入feature_size，输出3类别
        # # 添加分类头，输入维度为256，输出3类别
        # self.classifier = nn.Sequential(
        #     nn.Linear(feature_size, feature_size//4),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Linear(feature_size//4, 3)
        # )
        self.classifier = nn.Linear(feature_size, 3)

        self._init_weights()
        self.src_mask = None

    def _init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask=None):
        batch_size, seq_len, _ = src.size()
        src = self.input_projection(src)
        src = self.pos_encoder(src)

        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        output = self.transformer_encoder(src, src_mask)
        # 取最后一个时间步作为特征表示 [batch_size, feature_size]
        last_feature = output[:, -1, :]

        # 原始任务输出
        gen = self.decoder(last_feature)
        # 分类输出
        cls = self.classifier(last_feature)

        return gen, cls

    def _generate_square_subsequent_mask(self, seq_len):
        # 生成上三角掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

# RNN生成器模型
class Generator_rnn(nn.Module):
    def __init__(self, input_size):
        super(Generator_rnn, self).__init__()
        self.rnn_1 = nn.RNN(input_size, 1024, batch_first=True)
        self.rnn_2 = nn.RNN(1024, 512, batch_first=True)
        self.rnn_3 = nn.RNN(512, 256, batch_first=True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        use_cuda = 1
        device = x.device
        h0_1 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.rnn_1(x, h0_1)
        out_1 = self.dropout(out_1)
        h0_2 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.rnn_2(out_1, h0_2)
        out_2 = self.dropout(out_2)
        h0_3 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.rnn_3(out_2, h0_3)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out

class Discriminator3(nn.Module):
    def __init__(self, input_dim, out_size, hidden_dim):
        """
        input_dim: 每个时间步的特征数，比如你是21
        out_size: 你想输出几个预测值，比如5
        """
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim+1, hidden_dim//4, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, stride=1, padding='same')

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.batch1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_size)

        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, T, F] => [B, F, T]
        #x = x.permute(0, 2, 1)

        conv1 = self.leaky(self.conv1(x))  # [B, 32, T]
        conv2 = self.leaky(self.conv2(conv1))  # [B, 64, T]
        conv3 = self.leaky(self.conv3(conv2))  # [B, 128, T]

        # 聚合时间信息，取平均
        pooled = torch.mean(conv3, dim=2)  # [B, 128]

        out = self.leaky(self.linear1(pooled))  # [B, 220]
        out = self.relu(self.linear2(out))     # [B, 220]
        out = self.relu(self.linear3(out))  # [B, out_size]

        return out