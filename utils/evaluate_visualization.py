import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def validate(model, val_x, val_y):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁止计算梯度
        val_x = val_x.clone().detach().float()

        # 检查val_y的类型，如果是numpy.ndarray则转换为torch.Tensor
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        else:
            val_y = val_y.clone().detach().float()

        # 使用模型进行预测
        predictions, logits  = model(val_x)
        predictions = predictions.cpu().numpy()
        val_y = val_y.cpu().numpy()

        # 计算均方误差（MSE）作为验证损失
        mse_loss = F.mse_loss(torch.tensor(predictions).float().squeeze(), torch.tensor(val_y).float().squeeze())

        return mse_loss

def validate_with_label(model, val_x, val_y, val_labels):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁止计算梯度
        val_x = val_x.clone().detach().float()

        # 检查val_y的类型，如果是numpy.ndarray则转换为torch.Tensor
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        else:
            val_y = val_y.clone().detach().float()

        # labels 用于分类
        if isinstance(val_labels, np.ndarray):
            val_lbl_t = torch.tensor(val_labels).long().to(val_x.device)
        else:
            val_lbl_t = val_labels.clone().detach().long().to(val_x.device)

        # 使用模型进行预测
        predictions, logits  = model(val_x)
        predictions = predictions.cpu().numpy()
        val_y = val_y.cpu().numpy()

        # 计算均方误差（MSE）作为验证损失
        mse_loss = F.mse_loss(torch.tensor(predictions).float().squeeze(), torch.tensor(val_y).float().squeeze())

        true_cls = val_lbl_t[:, -1].squeeze()  # [B]
        pred_cls = logits.argmax(dim=1)  # [B]
        acc = (pred_cls == true_cls).float().mean()  # 标量

        return mse_loss, acc


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
    return scaler.inverse_transform(predictions)


def compute_metrics(true_values, predicted_values):
    """计算MSE, MAE, RMSE, MAPE"""
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    per_target_mse = np.mean((true_values - predicted_values) ** 2, axis=0)  # 新增
    return mse, mae, rmse, mape, per_target_mse


def plot_fitting_curve(true_values, predicted_values, output_dir, model_name,target_name):
    """绘制拟合曲线并保存结果"""
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 8))
    plt.plot(true_values, label='True Values', linewidth=2)
    plt.plot(predicted_values, label='Predicted Values', linewidth=2, linestyle='--')
    plt.title(f'{model_name} on {target_name}', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{target_name}_{model_name}_fitting_curve.png', dpi=500)
    plt.close()


def save_metrics(metrics, output_dir, model_name):
    """保存MSE, MAE, RMSE, MAPE到文件"""
    with open(f'{output_dir}/{model_name}_metrics.txt', 'w') as f:
        f.write("MSE: {}\n".format(metrics[0]))
        f.write("MAE: {}\n".format(metrics[1]))
        f.write("RMSE: {}\n".format(metrics[2]))
        f.write("MAPE: {}\n".format(metrics[3]))




def evaluate_best_models(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir,
                         generator_names,target_name,ERM):
    N = len(generators)

    for i in range(N):
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()

    train_y_inv = inverse_transform(train_y, y_scaler)
    test_y_inv = inverse_transform(test_y, y_scaler)


    train_results = []
    test_results = []

    with torch.no_grad():
        train_fitting_curve_dir=os.path.join(output_dir, "train")
        test_fitting_curve_dir=os.path.join(output_dir, "test")
        for i in range(N):
            if N>1:
                name = f'MAA-TSF-{generator_names[i]}'
            elif not ERM:
                name = f'GAN-{generator_names[i]}'
            else:
                name = f'ERM-{generator_names[i]}'

            # Train
            train_pred, _ = generators[i](train_xes[i])
            train_pred = train_pred.cpu().numpy()
            train_pred_inv = inverse_transform(train_pred, y_scaler)
            train_metrics = compute_metrics(train_y_inv, train_pred_inv)
            plot_fitting_curve(train_y_inv, train_pred_inv,train_fitting_curve_dir, name,target_name)

            # Test
            test_pred, _ = generators[i](test_xes[i])
            test_pred = test_pred.cpu().numpy()
            test_pred_inv = inverse_transform(test_pred, y_scaler)
            test_metrics = compute_metrics(test_y_inv, test_pred_inv)
            plot_fitting_curve(test_y_inv, test_pred_inv, test_fitting_curve_dir, name,target_name)

            # Logging
            print(f"[Train] {name}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, "
                  f"RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
            print(f"[Test]  {name}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, "
                  f"RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

            logging.info(f"[Train] {name}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, "
                         f"RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
            logging.info(f"[Test]  {name}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, "
                         f"RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

            # Collect for external use
            train_results.append({
                "Generator": name,
                "MSE": train_metrics[0],
                "MAE": train_metrics[1],
                "RMSE": train_metrics[2],
                "MAPE": train_metrics[3],
                "MSE_per_target": train_metrics[4].tolist()
            })

            test_results.append({
                "Generator": name,
                "MSE": test_metrics[0],
                "MAE": test_metrics[1],
                "RMSE": test_metrics[2],
                "MAPE": test_metrics[3],
                "MSE_per_target": test_metrics[4].tolist()
            })

    # 不直接保存 CSV，把 train/test results 留给外部处理
    return train_results, test_results


def plot_density(all_true_series, pred_series_list, pred_labels, output_dir, alpha, no_grid,mode,target_name):
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))

    # 合并所有真实值数据并绘制其总体的密度分布
    combined_true = pd.concat(all_true_series).dropna()
    if not combined_true.empty:
        sns.kdeplot(combined_true, label='True Value', color='orange',
                    linewidth=1.5, alpha=alpha, fill=True)
    else:
        print("⚠️ 未找到真实的有效数据，跳过绘制 True 分布。")

    print("正在绘制所有预测分布...")
    for pred_series, label in zip(pred_series_list, pred_labels):
        if not pred_series.empty:
            sns.kdeplot(pred_series, label=f'Predictions on {label}',
                        linewidth=1.5, alpha=alpha, fill=True)
        else:
            print(f"⚠️ 文件 {label} 中未找到预测的有效数据，跳过绘制。")

    ax = plt.gca()
    ax.set(xlabel='Value', ylabel='Density')
    plt.title(f'MAA-TSF on {target_name}', fontsize=16)
    plt.legend()

    if not no_grid:
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{target_name}_{mode}_density.png')
    try:
        plt.savefig(out_path)
        print(f"Saved combined plot: {out_path}")
    except Exception as e:
        print(f"❌ 无法保存图形 {out_path}: {e}")

    plt.close()

def read_and_collect_data(csv_paths):
    """
    读取所有 CSV 文件并收集数据
    Args:
        csv_paths (list): CSV 文件路径列表

    Returns:
        all_true_series (list): 真实值数据
        pred_series_list (list): 预测值数据
        pred_labels (list): 每个文件的标签
    """
    all_true_series = []
    pred_series_list = []
    pred_labels = []

    print("正在读取并收集数据...")

    for path in csv_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"❌ 无法读取文件 {path}: {e}")
            continue

        if 'true' not in df.columns or 'pred' not in df.columns:
            print(f"⚠️ 文件 {path} 中缺少 'true' 或 'pred' 列, 跳过。")
            continue

        all_true_series.append(df['true'].dropna())
        pred_series_list.append(df['pred'].dropna())
        pred_labels.append(filename)  # 使用文件名作为预测分布的标签

    if not all_true_series:
        print("❌ 未在任何文件中找到有效数据。")
        exit(1)

    return all_true_series, pred_series_list, pred_labels