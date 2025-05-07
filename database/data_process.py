import pandas as pd
import numpy as np
import argparse
import os


def calculate_metrics(input_pdframe, save_path, date_clip):
    # === 1. 读取文件 ===
    df = input_pdframe

    # 若存在时间列则设为索引，并根据date_clip筛选数据
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  # 确保'交易日期'是datetime类型
        df = df.set_index('time')  # 将 'time' 设置为索引

    # 确保 date_clip 中的日期是 datetime 类型
    start_date, end_date = pd.to_datetime(date_clip[0]), pd.to_datetime(date_clip[1])

    # 筛选日期范围
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    # === 2. 指标计算函数 ===
    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    def sma(series, window):
        return series.rolling(window).mean()

    # === 3. 计算指标（若不存在则新增） ===
    # 3.1 MA
    for w in (5, 15, 30):
        col = f"MA{w}"
        if col not in df.columns:
            df[col] = sma(df['close'], w)

    # 3.2 MACD
    if not {'MACD', 'DIF', 'DEA'}.issubset(df.columns):
        fast, slow, signal = 12, 26, 9
        ema_fast = ema(df['close'], fast)
        ema_slow = ema(df['close'], slow)
        df['DIF'] = ema_fast - ema_slow
        df['DEA'] = ema(df['DIF'], signal)
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])

    # 3.3 ATR
    if 'ATR' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

    # 3.4 Bollinger Bands
    if not {'BOLL_MID', 'BOLL_UP', 'BOLL_LOW'}.issubset(df.columns):
        window = 20
        mid = sma(df['close'], window)
        std = df['close'].rolling(window).std()
        df['BOLL_MID'] = mid
        df['BOLL_UP'] = mid + 2 * std
        df['BOLL_LOW'] = mid - 2 * std

    # 3.5 RSI
    if 'RSI' not in df.columns:
        period = 14
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=df.index).rolling(period).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # 3.6 KDJ
    if not {'K', 'D', 'J'}.issubset(df.columns):
        low_min = df['low'].rolling(9).min()
        high_max = df['high'].rolling(9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['K'] = rsv.ewm(alpha=1 / 3).mean()
        df['D'] = df['K'].ewm(alpha=1 / 3).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

    # === 4. 保留 6 位小数 ===
    df = df.round(6)

    # === 5. 删除空值 ===
    # df = df.dropna()

    # === 6. 保存并展示 ===
    output_path = save_path
    df.to_csv(output_path, float_format='%.6f')

    return df.tail()


# 使用 argparse 处理命令行参数
def main():
    parser = argparse.ArgumentParser(description="Calculate technical indicators and save the results.")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the processed CSV file.")
    parser.add_argument('--date_clip', type=str, nargs=2, required=True,
                        help="Start and end date for clipping data (format: 'YYYY-MM-DD').")

    args = parser.parse_args()

    # 读取 CSV 文件
    df = pd.read_csv(args.input_csv)

    # 处理数据并保存
    save_path = os.path.join(args.output_dir, os.path.basename(args.input_csv).replace('.csv', '_processed.csv'))
    save_path = save_path.replace('raw', 'processed')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 调用计算指标函数
    calculate_metrics(df, save_path, args.date_clip)

    print(f"处理结果已保存到: {save_path}")


if __name__ == "__main__":
    main()
