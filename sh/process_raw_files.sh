#!/bin/bash

# 数据文件目录
DATA_DIR="../database/raw"

# 输出目录
OUTPUT_DIR="../database/processed"

# 日期范围
START_DATE="2012-01-01"
END_DATE="2025-01-01"

# 确保该目录存在
if [ ! -d "$DATA_DIR" ]; then
    echo "目录 $DATA_DIR 不存在!"
    exit 1
fi

# 确保输出目录存在
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "输出目录 $OUTPUT_DIR 不存在, 创建中..."
    mkdir -p "$OUTPUT_DIR"
fi

# 遍历目录中的每个CSV文件
for FILE in "$DATA_DIR"/*.csv; do
    # 获取文件的 basename（去掉路径和扩展名）
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.csv}"  # 去掉 .csv 后缀

    # 打印当前文件的文件名
    echo "正在处理文件: $FILENAME"

    # 定义保存路径
    SAVE_PATH="$OUTPUT_DIR/${BASENAME}_processed.csv"

    # 调用 Python 脚本进行处理
    python ./database/data_process.py \
        --input_csv "$FILE" \
        --output_dir "$OUTPUT_DIR" \
        --date_clip "$START_DATE" "$END_DATE"

    echo "处理结果已保存到: $SAVE_PATH"
done

echo "所有文件处理完毕!"
