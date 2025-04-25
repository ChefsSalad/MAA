import matplotlib.pyplot as plt

# 全局美化样式设置
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

import numpy as np
import os

def plot_generator_losses(data_G, output_dir):
    plt.rcParams.update({'font.size': 12})
    all_data = data_G
    N = len(all_data)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(all_data):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            plt.plot(acc, label=f"G{i + 1} vs D{j + 1}" if j < len(data)-1 else f"G{i + 1} Combined", linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"G{i + 1} Loss over Epochs", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generator_losses.png"))
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
    plt.savefig(os.path.join(output_dir, "discriminator_losses.png"))
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
    plt.savefig(os.path.join(output_dir, "overall_losses.png"))
    plt.close()

def plot_mse_loss(hist_MSE_G, hist_val_loss, num_epochs, output_dir):
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
    plt.savefig(os.path.join(output_dir, "mse_losses.png"))
    plt.close()

def plot_fitting_curve(true_values, predicted_values, output_dir, model_name):
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
    plt.savefig(f'{output_dir}/{model_name}_fitting_curve.png')
    plt.close()


def generate_dummy_loss_data(num_epochs=20, num_models=3):
    """
    生成模拟的生成器/判别器损失数据
    返回格式: [[D1_loss_list, D2_loss_list, D3_loss_list, G_loss_list], ...]
    """
    all_losses = []
    for _ in range(num_models):
        D_losses = [np.random.rand(num_epochs).cumsum() / (np.arange(num_epochs)+1) for _ in range(num_models)]
        G_loss = np.random.rand(num_epochs).cumsum() / (np.arange(num_epochs)+1)
        all_losses.append(D_losses + [G_loss])
    return all_losses

def main():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = 30
    # 模拟生成器和判别器损失数据
    fake_generator_losses = generate_dummy_loss_data(num_epochs)
    fake_discriminator_losses = generate_dummy_loss_data(num_epochs)

    # 模拟MSE和验证MSE数据
    train_mse = [np.random.rand(num_epochs) * 0.05 for _ in range(3)]
    val_mse = [m + np.random.rand(num_epochs) * 0.02 for m in train_mse]

    # 模拟拟合曲线
    true_vals = np.linspace(0, 1, 100)
    pred_vals = true_vals + np.random.normal(0, 0.05, size=100)

    # 绘图函数调用
    plot_generator_losses(fake_generator_losses, output_dir)
    plot_discriminator_losses(fake_discriminator_losses, output_dir)
    visualize_overall_loss([g[-1] for g in fake_generator_losses], [d[-1] for d in fake_discriminator_losses], output_dir)
    plot_mse_loss(train_mse, val_mse, num_epochs, output_dir)
    plot_fitting_curve(true_vals, pred_vals, output_dir, model_name="DummyModel")

    print(f"✅ 所有图表已保存至: {output_dir}/")

if __name__ == "__main__":
    main()
