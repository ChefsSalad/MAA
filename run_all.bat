@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 设置数据文件路径
set DATA_DIR=D:\Desktop\SHU\Intern\同梁AI量化\papers\GCA_lite\database

REM 遍历文件列表
for %%F in (
    processed_300股指_day.csv
    processed_原油_day.csv
    processed_大豆_day.csv
    processed_白糖_day.csv
    processed_纸浆_day.csv
    processed_螺纹钢_day.csv
    processed_铜_day.csv
    processed_黄金_day.csv
) do (
    set "FILENAME=%%F"
    set "BASENAME=%%~nF"
    set "OUTPUT_DIR=output/!BASENAME!"

    REM 设置 start_timestamp
    set "START_TIMESTAMP=31"
    if "%%F"=="processed_原油_day.csv" set "START_TIMESTAMP=1546"
    if "%%F"=="processed_纸浆_day.csv" set "START_TIMESTAMP=1710"

    echo Running !FILENAME! with start=!START_TIMESTAMP!

    python38 run_multi_gan.py ^
        --num_epochs 1 ^
        --data_path "%DATA_DIR%\!FILENAME!" ^
        --output_dir "!OUTPUT_DIR!" ^
        --start_timestamp "!START_TIMESTAMP!"
)

endlocal
