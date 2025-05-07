import copy

from   utils.evaluate_visualization import *
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

import logging  # NEW


def train_baseframe(generator, dataloader,
                    y_scaler, train_x, train_y, val_x, val_y,
                    train_y_labels,val_y_labels,
                    num_epochs,
                    output_dir,
                    device,
                    logger=None):
   
    
    g_learning_rate = 2e-5

    # 二元交叉熵【损失函数，可能会有问题
    # criterion = nn.BCELoss()

    optimizers_G = torch.optim.AdamW(generator.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
                    
    # 为每个优化器设置 ReduceLROnPlateau 调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizers_G, mode='min', factor=0.1, patience=16, min_lr=1e-7)
                  
    best_epoch = -1

    # 定义生成历史记录的关键字
    """
    以三个为例，keys长得是这样得的：
    ['G1', 'G2', 'G3', 
    'D1', 'D2', 'D3', 
    'MSE_G1', 'MSE_G2', 'MSE_G3', 
    'val_G1', 'val_G2', 'val_G3', 
    'D1_G1', 'D2_G1', 'D3_G1', 'D1_G2', 'D2_G2', 'D3_G2', 'D1_G3', 'D2_G3', 'D3_G3'
    ]
    """

    keys = []
    g_keys = 'G1'
    MSE_g_keys = 'MSE_G1' 
    val_loss_keys = 'val_G1'

    keys.extend(g_keys)
    keys.extend(MSE_g_keys)
    keys.extend(val_loss_keys)

    # 创建包含每个值为np.zeros(num_epochs)的字典
    # hists_dict = {key: np.zeros(num_epochs) for key in keys}

    # best_mse = float('inf')
    best_loss = float("inf")
    best_acc = -1
    best_model_state = None

    patience_counter = 0
    patience = 50
    # feature_num = train_xes[0].shape[2]
    # target_num = train_y.shape[-1]
    print("start training")
    for epoch in range(num_epochs):
        # epo_start = time.time()
        generator.train()
        keys = []
        keys.extend(g_keys)
        keys.extend(MSE_g_keys)

        loss_dict = {key: [] for key in keys}

        for batch_idx, (x_last, y_last, label_last) in enumerate(dataloader):
            # TODO: maybe try to random select a gap from the whole time windows
            x_last = x_last.to(device)
            y_last = y_last.to(device)
            label_last = label_last.to(device)
            label_last = label_last.unsqueeze(-1)
            # print(x_last.shape, y_last.shape, label_last.shape)

            outputs = generator(x_last)
            fake_data_G, fake_data_cls = outputs


            cls_loss = F.cross_entropy(fake_data_cls, label_last[:, -1, :].long().squeeze())
            mse_loss = F.mse_loss(fake_data_G.squeeze(), y_last[:, -1, :].squeeze())
            total_loss = cls_loss + mse_loss
            
            optimizers_G.zero_grad()
            total_loss.backward()
            optimizers_G.step()

            scheduler.step(total_loss)

        #val_loss = validate(generator, val_x, val_y)
        train_loss, train_acc = validate_with_label(generator, train_x, train_y, train_y_labels)
        val_loss,val_acc = validate_with_label(generator, val_x, val_y, val_y_labels)
        # logging.info("EPOCH %3d | Train MSE: %.4f | Train Acc: %.2f%% | Val MSE: %.4f | Val Acc: %.2f%%",
        #              epoch + 1, train_loss.item(), train_acc.item() * 100, val_loss.item(), val_acc.item() * 100)
        logging.info("Epoch %3d | Train MSE: %.4f | Train Acc: %.4f | Val MSE: %.4f | Val Acc: %.4f",
                     epoch + 1, train_loss.item(), train_acc.item() , val_loss.item(), val_acc.item())

        if val_loss>best_loss:
            patience_counter +=1
            print(f'patience last: {patience - patience_counter}, best: {best_loss}, val: {val_loss}')
        else:
            patience_counter = 0
            best_model_state = copy.deepcopy(generator.state_dict())
            best_loss = val_loss

        if val_acc>best_acc:
            best_acc = val_acc

        if patience_counter > patience:
            break
    # results = evaluate_best_models([generator], [best_model_state], [train_x], train_y, [val_x], val_y, y_scaler,
    #                             output_dir)
    return [best_acc.item()], [best_model_state]

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    # 将当前文件所在目录的上级加入 sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
