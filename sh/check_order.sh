#!/bin/bash

# 运行脚本：run_multi_gan.py
# 遍历指定的CSV数据文件并设置对应参数

DATA_DIR="../database/kline_processed_data"


declare -A START_MAP
declare -A END_MAP
START_MAP["processed_原油_day.csv"]=1546
START_MAP["processed_纸浆_day.csv"]=1710
START_MAP["processed_PTA_day.csv"]=213
END_MAP["processed_PTA_day.csv"]=3120
START_MAP["processed_SP500_day.csv"]=1077
END_MAP["processed_SP500_day.csv"]=4284
END_MAP["processed_上期能源集运欧线_day.csv"]=332
START_MAP["processed_上证50_day.csv"]=1172
END_MAP["processed_上证50_day.csv"]=4270
END_MAP["processed_中国十年债_day.csv"]=2073
START_MAP["processed_中国大商所玉米sv_day.csv"]=1760
END_MAP["processed_中国大商所玉米sv_day.csv"]=4858
START_MAP["processed_中国天然橡胶_day.csv"]=546
END_MAP["processed_中国天然橡胶_day.csv"]=3651
START_MAP["processed_十年美债_day.csv"]=1608
END_MAP["processed_十年美债_day.csv"]=4938
END_MAP["processed_原木期货_day.csv"]=606
START_MAP["processed_原油期货_day.csv"]=1065
END_MAP["processed_原油期货_day.csv"]=4275
START_MAP["processed_普通小麦期货_day.csv"]=51
END_MAP["processed_普通小麦期货_day.csv"]=3057
START_MAP["processed_比特币_day.csv"]=820
END_MAP["processed_比特币_day.csv"]=5476
START_MAP["processed_沪深300_day.csv"]=649
END_MAP["processed_沪深300_day.csv"]=3747
END_MAP["processed_甲醇_day.csv"]=2504
START_MAP["processed_纸浆期货合约_day.csv"]=1073
END_MAP["processed_纸浆期货合约_day.csv"]=4280
START_MAP["processed_美元指数_day.csv"]=1060
END_MAP["processed_美元指数_day.csv"]=4284
START_MAP["processed_美大豆期货_day.csv"]=1087
END_MAP["processed_美大豆期货_day.csv"]=4297
START_MAP["processed_螺纹钢_day.csv"]=546
END_MAP["processed_螺纹钢_day.csv"]=3652
START_MAP["processed_道琼斯_day.csv"]=1077
END_MAP["processed_道琼斯_day.csv"]=4284


# 默认的 start_timestamp
DEFAULT_START=31
DEFAULT_END=-1

for FILE in "$DATA_DIR"/processed_*_day.csv; do
    FILENAME=$(basename "$FILE")
  
    
    # 设置输出目录（可按需更换）
  
    echo "Running $FILENAME"

        
done

