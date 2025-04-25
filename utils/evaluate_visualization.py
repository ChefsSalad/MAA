import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging


# 年化收益
def calculate_annualized_return(returns, periods_per_year=252):
    cumulative_return = np.prod([1 + r for r in returns]) - 1
    years = len(returns) / periods_per_year
    if years == 0:
        return 0
    annualized_return = (1 + cumulative_return) ** (1 / years) - 1
    return annualized_return * 100

# 夏普比率（默认无风险利率为0）
def calculate_sharpe_ratio(returns, periods_per_year=252):
    excess_returns = np.array(returns)
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    if std_excess == 0:
        return 0
    sharpe = (mean_excess * np.sqrt(periods_per_year)) / std_excess
    return sharpe

# 最大回撤
def calculate_max_drawdown(returns):
    cumulative_returns = np.cumprod([1 + r for r in returns])
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdowns)
    return abs(max_drawdown) * 100

# 胜率（可选）
def calculate_win_rate(returns):
    returns = np.array(returns)
    profitable_trades = np.sum(returns > 0)
    total_trades = np.sum(returns != 0)
    return (profitable_trades / total_trades) * 100 if total_trades > 0 else 0


def validate_financial_metric(generator, train_x,train_y, val_x, val_y,val_label_y,y_scaler):
    # 加载模型并设为 eval
    generator.eval()

    # 反向变换（逆变换）
    train_y_inv = inverse_transform(train_y, y_scaler)
    val_y_inv = inverse_transform(val_y, y_scaler)

    train_preds_inv = []
    val_preds_inv = []
    train_metrics_list = []
    val_metrics_list = []

    with torch.no_grad():

        # 获取训练和测试集的预测结果
        train_pred, train_cls = generator(train_x)
        test_pred, test_cls = generator(val_x)

        # 将预测转换为 numpy 数组并反向变换
        train_pred = train_pred.cpu().numpy()
        test_pred = test_pred.cpu().numpy()
        train_pred_inv = inverse_transform(train_pred, y_scaler)
        test_pred_inv = inverse_transform(test_pred, y_scaler)

        train_preds_inv.append(train_pred_inv)
        val_preds_inv.append(test_pred_inv)

        # 获取模型预测的动作
        train_pred_labels = train_cls.argmax(dim=-1).cpu().numpy()  # 获取训练集的预测动作
        test_pred_labels = test_cls.argmax(dim=-1).cpu().numpy()    # 获取测试集的预测动作

        # 获取真实的价格数据（这里假设已经有真实价格数据 real_train_prices 和 real_test_prices）
        real_train_prices = train_y_inv.flatten()
        real_test_prices = val_y_inv.flatten()

        # 计算回报（使用预测的动作和真实的价格）
        train_returns = []
        test_returns = []

        # 训练集回报计算
        for i in range(1, len(real_train_prices)):
            price_change = (real_train_prices[i] - real_train_prices[i-1] )/ real_train_prices[i - 1]
            action = train_pred_labels[i]
            if action == 1:  # 做多（Buy）
                train_returns.append(price_change)
            elif action == 2:  # 做空（Sell）
                train_returns.append(-price_change)
            else:  # 持有（Hold）
                train_returns.append(0)

        # 测试集回报计算
        for i in range(1, len(real_test_prices)):
            price_change = (real_test_prices[i] - real_test_prices[i-1])/ real_test_prices[i - 1]
            action = test_pred_labels[i]
            #0 = hold, 1 = buy, 2 = sell
            if action == 1:  # 做多（Buy）
                test_returns.append(price_change)
            elif action == 2:  # 做空（Sell）
                test_returns.append(-price_change)
            else:  # 持有（Hold）
                test_returns.append(0)

        # 计算财务指标
        periods_per_year = 252  # 如果是日频数据（你的是）
        print("train days:",len(train_returns))
        print("test days:",len(test_returns))
        train_sharpe_ratio = calculate_sharpe_ratio(train_returns, periods_per_year)
        test_sharpe_ratio = calculate_sharpe_ratio(test_returns, periods_per_year)

        train_max_drawdown = calculate_max_drawdown(train_returns)
        test_max_drawdown = calculate_max_drawdown(test_returns)

        train_annualized_return = calculate_annualized_return(train_returns, periods_per_year)
        test_annualized_return = calculate_annualized_return(test_returns, periods_per_year)

        train_win_rate = calculate_win_rate(train_returns)
        test_win_rate = calculate_win_rate(test_returns)

        # 记录训练集指标
        train_metrics_list.append({
            'sharpe_ratio': round(train_sharpe_ratio, 4),
            'max_drawdown': round(train_max_drawdown, 2),
            'annualized_return': round(train_annualized_return, 2),
            'win_rate': round(train_win_rate, 2)
        })

        # 记录验证/测试集指标
        val_metrics_list.append({
            'sharpe_ratio': round(test_sharpe_ratio, 4),
            'max_drawdown': round(test_max_drawdown, 2),
            'annualized_return': round(test_annualized_return, 2),
            'win_rate': round(test_win_rate, 2)
        })

        # 返回完整的指标列表
        print_metrics(train_metrics_list, val_metrics_list)
        return train_metrics_list, val_metrics_list




def validate(model, val_x, val_y, val_label_y, predict_step=1, device='cuda'):
    model.eval()
    model = model.to(device)  # ✅ 模型移到 device
    with torch.no_grad():
        # ✅ 所有数据都移到 device
        val_x = val_x.clone().detach().float().to(device)

        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        val_y = val_y.clone().detach().float().to(device)

        if isinstance(val_label_y, np.ndarray):
            val_label_y = torch.tensor(val_label_y).long()
        val_label_y = val_label_y.clone().detach().long().to(device)

        # 模型预测
        reg_preds, cls_preds = model(val_x)

        # 预测 & 标签都在 device 上，此处无需再转
        reg_preds = reg_preds[:, -predict_step:]


        val_y = val_y[:, -predict_step:].squeeze(-1)
        if predict_step > 1:
            val_y = val_y.squeeze(dim=1)
        mse_loss = F.mse_loss(reg_preds, val_y)

        cls_preds = cls_preds[:, -predict_step:, :]
        cls_targets = val_label_y[:, -predict_step:]

        pred_labels = cls_preds.argmax(dim=-1)
        pred_labels = pred_labels.long()
        cls_targets = cls_targets.long()

        #print(pred_labels.shape, cls_targets.shape,cls_targets.numel())
        acc = (pred_labels == cls_targets).sum() / (cls_targets.numel())

    return mse_loss.item(), acc.item()


def print_metrics(train_metrics_list, val_metrics_list):
    print("📊 回测结果指标（每轮）")
    print("=" * 40)
    for i, (train_metrics, val_metrics) in enumerate(zip(train_metrics_list, val_metrics_list)):
        print("  📘 训练集:")
        print(f"    Sharpe Ratio       : {train_metrics['sharpe_ratio']}")
        print(f"    Max Drawdown       : {train_metrics['max_drawdown']}%")
        print(f"    Annualized Return  : {train_metrics['annualized_return']}%")
        print(f"    Win Rate           : {train_metrics.get('win_rate', 'N/A')}%")
        print("  📕 测试集:")
        print(f"    Sharpe Ratio       : {val_metrics['sharpe_ratio']}")
        print(f"    Max Drawdown       : {val_metrics['max_drawdown']}%")
        print(f"    Annualized Return  : {val_metrics['annualized_return']}%")
        print(f"    Win Rate           : {val_metrics.get('win_rate', 'N/A')}%")
        print("-" * 40)


def plot_generator_losses(data_G, output_dir):
    """
    绘制 G1、G2、G3 的损失曲线。

    Args:
        data_G1 (list): G1 的损失数据列表，包含 [histD1_G1, histD2_G1, histD3_G1, histG1]。
        data_G2 (list): G2 的损失数据列表，包含 [histD1_G2, histD2_G2, histD3_G2, histG2]。
        data_G3 (list): G3 的损失数据列表，包含 [histD1_G3, histD2_G3, histD3_G3, histG3]。
    """

    plt.rcParams.update({'font.size': 12})
    all_data = data_G
    N = len(all_data)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(all_data):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            plt.plot(acc, label=f"G{i + 1} vs D{j + 1}" if j < N - 1 else f"G{i + 1} Combined", linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"G{i + 1} Loss over Epochs", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generator_losses.png"), dpi=500)
    plt.close()


def plot_discriminator_losses(data_D, output_dir):
    plt.rcParams.update({'font.size': 12})
    N = len(data_D)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(data_D):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            plt.plot(acc, label=f"D{i + 1} vs G{j + 1}" if j < len(data)-1 else f"D{i + 1} Combined", linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"D{i + 1} Loss over Epochs", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "discriminator_losses.png"), dpi=500)
    plt.close()


def visualize_overall_loss(histG, histD, output_dir):
    plt.rcParams.update({'font.size': 12})
    N = len(histG)
    plt.figure(figsize=(5 * N, 4))

    for i, (g, d) in enumerate(zip(histG, histD)):
        plt.plot(g, label=f"G{i + 1} Loss", linewidth=2)
        plt.plot(d, label=f"D{i + 1} Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Generator & Discriminator Loss", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_losses.png"), dpi=500)
    plt.close()


def plot_mse_loss(hist_MSE_G, hist_val_loss, num_epochs,
                  output_dir):
    """
    绘制训练过程中和验证集上的MSE损失变化曲线

    参数：
    hist_MSE_G1, hist_MSE_G2, hist_MSE_G3 : 训练过程中各生成器的MSE损失
    hist_val_loss1, hist_val_loss2, hist_val_loss3 : 验证集上各生成器的MSE损失
    num_epochs : 训练的epoch数
    """
    plt.rcParams.update({'font.size': 12})
    N = len(hist_MSE_G)
    plt.figure(figsize=(5 * N, 4))

    for i, (MSE, val_loss) in enumerate(zip(hist_MSE_G, hist_val_loss)):
        plt.plot(range(num_epochs), MSE, label=f"Train MSE G{i + 1}", linewidth=2)
        plt.plot(range(num_epochs), val_loss, label=f"Val MSE G{i + 1}", linewidth=2, linestyle="--")

    plt.title("MSE Loss for Generators (Train vs Validation)", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_losses.png"), dpi=500)
    plt.close()

def inverse_transform(predictions, scaler):
    """ 使用y_scaler逆转换预测结果 """
    #print(predictions.shape)
    original_shape = predictions.shape
    reshaped = predictions.reshape(-1, original_shape[-1])  # (batch * steps, 1)
    restored = scaler.inverse_transform(reshaped)
    return restored
    #return scaler.inverse_transform(predictions)


def compute_metrics(true_values, predicted_values):
    """计算MSE, MAE, RMSE, MAPE"""
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    per_target_mse = np.mean((true_values - predicted_values) ** 2, axis=0)  # 新增
    return mse, mae, rmse, mape, per_target_mse


def plot_fitting_curve(true_values, predicted_values, output_dir, model_name):
    """绘制拟合曲线并保存结果"""
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='True Values', linewidth=2)
    plt.plot(predicted_values, label='Predicted Values', linewidth=2, linestyle='--')
    plt.title(f'{model_name} Fitting Curve', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_fitting_curve.png', dpi=500)
    plt.close()


def save_metrics(metrics, output_dir, model_name):
    """保存MSE, MAE, RMSE, MAPE到文件"""
    with open(f'{output_dir}/{model_name}_metrics.txt', 'w') as f:
        f.write("MSE: {}\n".format(metrics[0]))
        f.write("MAE: {}\n".format(metrics[1]))
        f.write("RMSE: {}\n".format(metrics[2]))
        f.write("MAPE: {}\n".format(metrics[3]))


def evaluate_best_models(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir):
    N = len(generators)

    # 加载模型并设为 eval
    for i in range(N):
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()

    train_y_inv = inverse_transform(train_y, y_scaler)
    test_y_inv = inverse_transform(test_y, y_scaler)

    train_preds_inv = []
    test_preds_inv = []
    train_metrics_list = []
    test_metrics_list = []

    with torch.no_grad():
        for i in range(N):
            train_pred, train_cls = generators[i](train_xes[i])
            train_pred = train_pred.cpu().numpy()
            train_pred_inv = inverse_transform(train_pred, y_scaler)
            train_preds_inv.append(train_pred_inv)
            train_metrics = compute_metrics(train_y_inv, train_pred_inv)
            train_metrics_list.append(train_metrics)
            plot_fitting_curve(train_y_inv, train_pred_inv, output_dir, f'G{i+1}_Train')
            print(f"Train Metrics for G{i+1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
            logging.info(f"Train Metrics for G{i+1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")

        for i in range(N):
            test_pred, test_cls = generators[i](test_xes[i])
            test_pred = test_pred.cpu().numpy()
            test_pred_inv = inverse_transform(test_pred, y_scaler)
            test_preds_inv.append(test_pred_inv)
            test_metrics = compute_metrics(test_y_inv, test_pred_inv)
            test_metrics_list.append(test_metrics)
            plot_fitting_curve(test_y_inv, test_pred_inv, output_dir, f'G{i+1}_Test')
            print(f"Test Metrics for G{i+1}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")
            logging.info(f"Test Metrics for G{i+1}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

    # 构造返回结果
    result = {
        "train_mse":  [m[0] for m in train_metrics_list],
        "train_mae":  [m[1] for m in train_metrics_list],
        "train_rmse": [m[2] for m in train_metrics_list],
        "train_mape": [m[3] for m in train_metrics_list],
        "train_mse_per_target": [m[4] for m in train_metrics_list],

        "test_mse":  [m[0] for m in test_metrics_list],
        "test_mae":  [m[1] for m in test_metrics_list],
        "test_rmse": [m[2] for m in test_metrics_list],
        "test_mape": [m[3] for m in test_metrics_list],
        "test_mse_per_target": [m[4] for m in test_metrics_list],
    }

    return result

