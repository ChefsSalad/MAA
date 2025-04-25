#!/bin/bash

# 运行脚本：run_multi_gan.py
# 遍历指定的CSV数据文件并设置对应参数

DATA_DIR="./database"

declare -A START_MAP
START_MAP["processed_原油_day.csv"]=1546
START_MAP["processed_纸浆_day.csv"]=1710

# 默认的 start_timestamp
DEFAULT_START=31

for FILE in "$DATA_DIR"/processed_*_day.csv; do
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.csv}"

    # 设置输出目录（可按需更换）
    OUTPUT_DIR="./output/${BASENAME}"

    # 判断是否在特殊映射中
    if [[ -v START_MAP["$FILENAME"] ]]; then
        START_TIMESTAMP=${START_MAP["$FILENAME"]}
    else
        START_TIMESTAMP=$DEFAULT_START
    fi

    echo "Running $FILENAME with start=$START_TIMESTAMP..."

    python run_multi_gan.py \
        --data_path "$FILE" \
        --output_dir "$OUTPUT_DIR" \
        --start_timestamp "$START_TIMESTAMP"
done

