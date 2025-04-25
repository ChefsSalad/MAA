import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer  # Transformer 模块
import math
import numpy as np
import torch.nn.functional as F

#滑动窗口的值设置为3

#变分自编码器VAE
class VAE(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        for i in range(1, len(config)):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i - 1], config[i]),
                    nn.ReLU()
                )
            )
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, config[-1])

        for i in range(len(config) - 1, 1, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i], config[i - 1]),
                    nn.ReLU()
                )
            )       
        modules.append(
            nn.Sequential(
                nn.Linear(config[1], config[0]),
                nn.Sigmoid()
            )
        ) 

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    def decode(self, x):
        result = self.decoder(x)
        return result

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5* logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar    
#滑动窗口方法代码
def sliding_window(x, y, window):
    x_ = []
    y_ = []
    y_gan = []
    for i in range(window, x.shape[0]):
        tmp_x = x[i - window: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window: i + 1]
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_ = torch.from_numpy(np.array(x_)).float()
    y_ = torch.from_numpy(np.array(y_)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_, y_, y_gan
#gru生成器模型
'''
class Generator_gru(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru_1 = nn.GRU(input_size, 1024, batch_first = True)
        self.gru_2 = nn.GRU(1024, 512, batch_first = True)
        self.gru_3 = nn.GRU(512, 256, batch_first = True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        use_cuda = 1
        device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out
'''

class Generator_gru(nn.Module):
    def __init__(self, input_size,out_size):
        super().__init__()
        self.gru = nn.GRU(input_size, 256, batch_first=True)  # 仅保留一层 GRU，隐藏维度设为 256
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, out_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        device = x.device
        
        # 初始化 GRU 的隐藏状态
        h0 = torch.zeros(1, x.size(0), 256).to(device)

        # 通过 GRU 层
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])  # 取序列最后一个时间步的输出
        
        # 通过全连接层
        out = self.linear_1(out)
        out = self.linear_2(out)
        out = self.linear_3(out)

        return out        
#lstm生成器模型
class Generator_lstm(nn.Module):
    def __init__(self, input_size,out_size):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=input_size, out_channels= input_size * 8, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=input_size * 8, hidden_size=128,
                         num_layers=2, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(in_features=128, out_features=out_size)

    def forward(self, x, hidden=None):
        x = x.permute(0, 2, 1)
        cnn_out = nn.LeakyReLU()(self.cnn(x))
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, hidden = self.lstm(cnn_out, hidden)
        pooled_out = F.adaptive_avg_pool1d(lstm_out.permute(0, 2, 1), 1)  # 输出 (128, 128, 1)
        linear_out = self.linear(pooled_out.squeeze(2))
        # linear_out = self.linear(lstm_out)
        return linear_out

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
# Transformer生成器模型
class Generator_transformer(nn.Module):
    def __init__(self, input_dim, feature_size=128, num_layers=2, num_heads=8, dropout=0.1, output_len=1):
        """
        input_dim: 数据特征维度
        feature_size: 模型特征维度
        num_layers: 编码器层数
        num_heads: 注意力头数目
        dropout: dropout权重
        output_len: 预测时间步长度
        """
        super().__init__()
        self.feature_size = feature_size
        self.output_len = output_len
        self.input_projection = nn.Linear(input_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        # 修改这里：添加 batch_first=True
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, output_len)  # 支持长时间步预测
        self._init_weights()
        self.src_mask = None

    def _init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask=None):
        batch_size, seq_len, _ = src.size()  # [batch_size, seq_len, d_model]

        src = self.input_projection(src)
        src = self.pos_encoder(src)

        # 这里不需要 permute，因为 batch_first=True
        # src = src.permute(1, 0, 2)  # [seq_len, batch_size, d_model]

        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        output = self.transformer_encoder(src, src_mask)
        last_step = output[:, -1, :]  # 修改这里，因为 batch_first=True, 现在batch_size 是第一维度, 形状 [batch_size, d_model]

        # 再映射到 [batch_size, output_len]
        last_step = self.decoder(last_step)

        return last_step

    def _generate_square_subsequent_mask(self, seq_len):
        """
        只关注时间步维度的掩码: [seq_len, seq_len]
        """
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

#CNN判别器模型
class Discriminator1(nn.Module):
    def __init__(self, window_size1,out_size):
        #采用3层卷积层（Conv1d）提取特征，并通过全连接层输出真假分类。
        super().__init__()
        self.conv1 = nn.Conv1d(window_size1+1, 32, kernel_size = 3, stride = 1, padding = 'same')#需要根据窗口的大小来调整第一个参数，值为窗口大小加一
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 3, stride = 1, padding = 'same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 3, stride = 1, padding = 'same')
        self.linear1 = nn.Linear(128, 220)
        self.batch1 = nn.BatchNorm1d(220)
        self.linear2 = nn.Linear(220, 220)
        self.batch2 = nn.BatchNorm1d(220)
        self.linear3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x = conv3.reshape(conv3.shape[0], conv3.shape[1])
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out_3 = self.linear3(out_2)
        out = self.sigmoid(out_3)
        return out

class Discriminator2(nn.Module):
    def __init__(self,window_size2,out_size):
        #采用3层卷积层（Conv1d）提取特征，并通过全连接层输出真假分类。
        super().__init__()
        self.conv1 = nn.Conv1d(window_size2+1, 32, kernel_size = 3, stride = 1, padding = 'same')#需要根据窗口的大小来调整第一个参数，值为窗口大小加一
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 3, stride = 1, padding = 'same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size = 3, stride = 1, padding = 'same')
        self.linear1 = nn.Linear(128, 220)
        self.batch1 = nn.BatchNorm1d(220)
        self.linear2 = nn.Linear(220, 220)
        self.batch2 = nn.BatchNorm1d(220)
        self.linear3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.leaky(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.leaky(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.leaky(conv3)
        flatten_x = conv3.reshape(conv3.shape[0], conv3.shape[1])
        out_1 = self.linear1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out_3 = self.linear3(out_2)
        out = self.sigmoid(out_3)
        return out
import torch
import torch.nn as nn

class Discriminator3(nn.Module):
    def __init__(self, input_dim, out_size):
        """
        Args:
            input_dim: 每个时间步的原始特征数（例如21）
            out_size: 输出预测值个数（例如5），通常用于预测真实/伪造打分或者分类
        """
        super().__init__()
        # 输入注意：原来的输入是 input_dim+1，这里保持不变，如实际数据需要可调整
        in_channels = input_dim + 1

        # 定义卷积层（1d卷积采用padding=1保证时序长度不变）
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # 自适应池化，将时间步维度池化为1（类似全局平均池化）
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层构成判别输出
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, out_size)

        # 激活函数统一采用 LeakyReLU
        self.activation = nn.LeakyReLU(0.01)
        # 最后输出层采用 Sigmoid（或者根据任务换成其他激活，比如不做激活直接输出打分）
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        输入：
            x: [B, T, F]，其中 T 为序列长度、F为特征数，此处 F 应为 input_dim+1
        输出：
            out: [B, out_size]
        """
        # 转换为 [B, F, T]，便于使用 Conv1d（因为 Conv1d 的输入通道一般放在第二维）
        x = x.permute(0, 2, 1)  # [B, F, T]

        out = self.activation(self.bn1(self.conv1(x)))  # [B, 32, T]
        out = self.activation(self.bn2(self.conv2(out)))  # [B, 64, T]
        out = self.activation(self.bn3(self.conv3(out)))  # [B, 128, T]

        # 自适应平均池化，将时间维度池化为1
        out = self.pool(out)  # [B, 128, 1]
        out = out.squeeze(-1)  # [B, 128]

        out = self.activation(self.fc1(out))  # [B, 128]
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))  # [B, out_size]

        return out

# #多头CNN自注意力机制判别器模型,有自注意力机制和全连接层，但是效果有点奇怪呀，一会试一下，判别器的模型优化器只使用一个，而不是使用4个优化器
# class MultiBranchDiscriminator1(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # 定义3个分支的卷积和全连接层
#         self.branch1 = self._create_branch(window_size1 + 1)
#         self.branch2 = self._create_branch(window_size2 + 1)
#         self.branch3 = self._create_branch(window_size3 + 1)
#
#         # 自注意力机制
#         self.attention_layer = nn.MultiheadAttention(embed_dim=3, num_heads=1, batch_first=True)
#
#         # 综合判别的全连接层
#         self.final_fc = nn.Linear(3, 1)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def _create_branch(self, input_channels):
#         """创建一个单独的判别器分支"""
#         return nn.Sequential(
#             nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.Flatten(),
#             nn.Linear(128, 220),
#             nn.LeakyReLU(0.01),
#             nn.Linear(220, 220),
#             nn.ReLU(),
#             nn.Linear(220, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x1, x2, x3):
#         # 通过三个分支
#         out1 = self.branch1(x1)  # (batch, 1)
#         out2 = self.branch2(x2)  # (batch, 1)
#         out3 = self.branch3(x3)  # (batch, 1)
#
#         # 拼接判别结果 (batch, 3)
#         outputs = torch.cat([out1, out2, out3], dim=1).unsqueeze(1)  # (batch, 1, 3)
#
#         # 通过自注意力机制，融合三个输出
#         attn_out, _ = self.attention_layer(outputs, outputs, outputs)  # (batch, 1, 3)
#         attn_out = attn_out.squeeze(1)  # (batch, 3)
#
#         # 通过全连接层计算最终判别结果
#         final_out = self.final_fc(attn_out)  # (batch, 1)
#         final_out = self.sigmoid(final_out)
#
#         return out1, out2, out3, final_out  # 返回三个单独结果和综合结果
# #最后的output-final是前三个的平均值，简单但是可以试一下效果
# class MultiBranchDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # 定义3个分支的卷积和全连接层
#         self.branch1 = self._create_branch(window_size1 + 1)
#         self.branch2 = self._create_branch(window_size2 + 1)
#         self.branch3 = self._create_branch(window_size3 + 1)
#
#
#         self.sigmoid = nn.Sigmoid()
#
#     def _create_branch(self, input_channels):
#         """创建一个单独的判别器分支"""
#         return nn.Sequential(
#             nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
#             nn.LeakyReLU(0.01),
#             nn.Flatten(),
#             nn.Linear(128, 220),
#             nn.LeakyReLU(0.01),
#             nn.Linear(220, 220),
#             nn.ReLU(),
#             nn.Linear(220, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x1, x2, x3):
#         # 通过三个分支
#         out1 = self.branch1(x1)  # (batch, 1)
#         out2 = self.branch2(x2)  # (batch, 1)
#         out3 = self.branch3(x3)  # (batch, 1)
#
#         # 计算平均值作为 final_out
#         final_out = (out1 + out2 + out3) / 3  # 平均结果
#
#         return out1, out2, out3, final_out.squeeze(1)  # 返回三个单独结果和综合结果
