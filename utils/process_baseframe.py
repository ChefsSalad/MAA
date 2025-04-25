import os
import re
import csv
import glob

# --- 配置 ---
BASE_DIR = '../output/baseframe'  # 日志文件的根目录
LOGS_SUBDIR = 'logs'      # 存放日志文件的子目录名
OUTPUT_CSV = 'consolidated_test_metrics.csv' # 输出的CSV文件名
# --- 配置结束 ---

def parse_metrics_line(line):
    """从日志行中解析指标值"""
    # 使用正则表达式查找指标=数值对
    # 这个模式假设指标按 MSE, MAE, RMSE, MAPE 的顺序出现，并允许中间有其他字符
    pattern = re.compile(
        r"MSE=([\d\.]+).*"
        r"MAE=([\d\.]+).*"
        r"RMSE=([\d\.]+).*"
        r"MAPE=([\d\.]+)"
    )
    match = pattern.search(line)
    if match:
        # 返回一个包含指标值的字典
        return {
            'MSE': match.group(1),
            'MAE': match.group(2),
            'RMSE': match.group(3),
            'MAPE': match.group(4),
        }
    else:
        return None

def find_and_process_logs(base_dir, logs_subdir, output_csv):
    """查找日志文件，处理它们，并生成CSV"""
    results = []
    # 构建查找日志文件的模式
    # e.g., ./baseframe/*/ */logs/*.log
    log_pattern = os.path.join(base_dir, '*', '*', logs_subdir, '*.log')
    log_files = glob.glob(log_pattern)

    print(f"找到 {len(log_files)} 个日志文件进行处理...")

    for log_file_path in log_files:
        try:
            # 从路径中提取 数据集名称 和 模型名称
            # 路径格式: ./baseframe/{数据集名称}/{模型名称}/logs/{文件名}.log
            parts = log_file_path.split(os.sep)
            if len(parts) >= 5 and parts[-2] == logs_subdir:
                model_name = parts[-3]
                dataset_name = parts[-4]
            else:
                print(f"警告: 无法从路径解析数据集/模型名称: {log_file_path}")
                continue

            last_test_line = None
            # 读取文件并找到最后包含 "Test Metrics" 的行
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 更精确地匹配，确保是测试指标行
                    if "INFO" in line and "Test Metrics" in line:
                        last_test_line = line.strip() # 保存找到的最后一行

            if last_test_line:
                metrics = parse_metrics_line(last_test_line)
                if metrics:
                    # 将数据集和模型名称添加到结果中
                    metrics['Dataset'] = dataset_name
                    metrics['Model'] = model_name
                    results.append(metrics)
                else:
                    print(f"警告: 无法解析指标行: {last_test_line} (来自文件: {log_file_path})")
            else:
                 print(f"警告: 在文件中未找到 'Test Metrics' 行: {log_file_path}")

        except FileNotFoundError:
            print(f"错误: 文件未找到: {log_file_path}")
        except Exception as e:
            print(f"处理文件时发生错误 {log_file_path}: {e}")

    # 写入CSV文件
    if not results:
        print("没有找到可写入CSV的结果。")
        return

    # 定义CSV表头顺序
    headers = ['Dataset', 'Model', 'MSE', 'MAE', 'RMSE', 'MAPE']

    print(f"正在将 {len(results)} 条结果写入 {output_csv}...")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader() # 写入表头
            writer.writerows(results) # 写入所有结果行
        print(f"成功创建CSV文件: {output_csv}")
    except IOError as e:
        print(f"写入CSV文件时出错 {output_csv}: {e}")
    except Exception as e:
         print(f"写入CSV时发生未知错误: {e}")

# --- 执行脚本 ---
if __name__ == "__main__":
    find_and_process_logs(BASE_DIR, LOGS_SUBDIR, OUTPUT_CSV)