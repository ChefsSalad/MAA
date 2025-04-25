import os
import pandas as pd
import ast
import re

def parse_any_list(s):
    """通用解析函数，处理以下格式：
    1. 标准列表 [val1, val2, val3]
    2. array格式 [array([val1]), array([val2]), array([val3])]
    3. np.float64格式 [np.float64(val1), np.float64(val2), np.float64(val3)]
    """
    try:
        # 先尝试直接解析标准列表
        try:
            return ast.literal_eval(s.strip())
        except:
            pass
        
        # 处理array格式
        array_matches = re.findall(r'array\(\[([\d.]+)\]\)', s)
        if array_matches:
            return [float(x) for x in array_matches]
        
        # 处理np.float64格式
        float64_matches = re.findall(r'np\.float64\(([\d.]+)\)', s)
        if float64_matches:
            return [float(x) for x in float64_matches]
        
        # 处理混合格式
        mixed_matches = re.findall(r'(?:array\(\[([\d.]+)\]\)|np\.float64\(([\d.]+)\)|([\d.]+))', s)
        if mixed_matches:
            values = []
            for match in mixed_matches:
                # match是三元组，取第一个非空的值
                val = next(x for x in match if x)
                values.append(float(val))
            return values
        
        print(f"无法解析的格式: {s}")
        return []
    except Exception as e:
        print(f"解析错误: {s} (错误: {str(e)})")
        return []

# 获取所有子文件夹
folders = [f for f in os.listdir() if os.path.isdir(f) and not f.startswith('.')]

results = []

for folder in folders:
    file_path = os.path.join(folder, "gca_GT_NPDC_market.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            
            # 解析所有列
            test_mse = parse_any_list(df['test_mse'].iloc[0])
            test_mae = parse_any_list(df['test_mae'].iloc[0])
            test_rmse = parse_any_list(df['test_rmse'].iloc[0])
            test_mape = parse_any_list(df['test_mape'].iloc[0])
            test_mse_per_target = parse_any_list(df['test_mse_per_target'].iloc[0])
            
            # 验证数据长度
            lengths = {
                'test_mse': len(test_mse),
                'test_mae': len(test_mae),
                'test_rmse': len(test_rmse),
                'test_mape': len(test_mape),
                'test_mse_per_target': len(test_mse_per_target)
            }
            
            if not all(l == 3 for l in lengths.values()):
                print(f"警告: {folder} 中的数据长度不一致: {lengths}")
                continue
            
            # 为每个模型创建一行
            models = ['GRU','LSTM', 'Transformer']
            for i, model in enumerate(models):
                try:
                    results.append({
                        'folder': folder,
                        'model': model,
                        'test_mse': test_mse[i],
                        'test_mae': test_mae[i],
                        'test_rmse': test_rmse[i],
                        'test_mape': test_mape[i],
                        'test_mse_per_target': test_mse_per_target[i]
                    })
                except IndexError:
                    print(f"警告: {folder} 中模型 {model} 的数据缺失")
                    continue
                
        except Exception as e:
            print(f"处理 {folder} 时出错: {str(e)}")
            continue

# 保存结果
if results:
    final_df = pd.DataFrame(results)
    # 按文件夹和模型排序
    final_df = final_df.sort_values(by=['folder', 'model'])
    
    # 使用UTF-8编码导出，确保正确处理中文等特殊字符
    final_df.to_csv('merged_results.csv', 
                   index=False, 
                   encoding='utf-8-sig',  # 使用带BOM的UTF-8以便Excel正确识别
                   float_format='%.8f')  # 控制浮点数精度
    
    print(f"\n合并完成，共处理 {len(folders)} 个文件夹")
    print(f"成功解析 {len(results)} 条记录 (预期 {len(folders)*3} 条)")
    print("UTF-8编码的结果已保存到 merged_results.csv")
    
    # 打印未成功处理的文件夹
    processed_folders = set(df['folder'] for df in results)
    failed_folders = set(folders) - processed_folders
    if failed_folders:
        print("\n以下文件夹处理失败:")
        for f in sorted(failed_folders):
            print(f" - {f}")
else:
    print("没有找到有效数据")