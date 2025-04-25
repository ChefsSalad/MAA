import pandas as pd
import numpy as np
import pymysql


def calculate_metrics(input_pdframe, save_path):
    # === 1. 读取文件 ===
    df = input_pdframe

    # 若存在时间列则设为索引
    if 'trade_date' in df.columns and not np.issubdtype(df['trade_date'].dtype, np.datetime64):
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')

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
        df['K'] = rsv.ewm(alpha=1/3).mean()
        df['D'] = df['K'].ewm(alpha=1/3).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

    # === 4. 保留 6 位小数 ===
    df = df.round(6)

    # === 5. 保存并展示 ===
    output_path = save_path
    df.to_csv(output_path, float_format='%.6f')

    df.tail()


actual_names = ['300股指','大豆','螺纹钢','黄金','铜','原油','纸浆','白糖']
# 300股指,大豆,螺纹钢,黄金,铜,原油,纸浆,白糖
future_codes = ['8600', '1300', '6880', '2270', '2100', '2460', '6220', '1840']
target_table_names = [fcode+'_day' for fcode in future_codes]
# trade_date,open,high,low,close,vol,
col_names_to_extract = ['trade_date', 'open', 'high', 'low', 'close', 'vol']
date_clip = ['20120101', '20250101']

# 数据库连接信息
db_config = {
    'host': '180.76.152.14',
    'port': 3306,
    'user': 'future',
    'password': 'tl1009',
    'database': 'future'
}

# 连接到数据库并读取数据
try:
    connection = pymysql.connect(**db_config)
    print("成功连接到数据库！")

    for actual_name, table_name in zip(actual_names, target_table_names):
        print(f'processing table: {table_name}')
        # 从 table 中 按时间截取 columns
        # 使用 SQL 查询读取数据
        query = f'SELECT {", ".join(col_names_to_extract)} FROM {table_name} WHERE trade_date BETWEEN "{date_clip[0]}" AND "{date_clip[1]}"'
        df = pd.read_sql(query, connection)
        column_count = df.shape[1]  # 统计列的数量
        print(f'当前数据框的列数为: {column_count}')
        calculate_metrics(df, './'+f'processed_{actual_name}_day.csv')

finally:
    connection.close()
    print("数据库连接已关闭。")
