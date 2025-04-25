@ECHO OFF
SETLOCAL EnableDelayedExpansion

REM # 运行脚本: run_baseframe.py
REM # 遍历指定的CSV数据文件
REM # 遍历指定的 generator 和对应的 window_size 组合
REM # 输出目录结构: .\output\<dataset_basename>\<generator_name>

SET "DATA_DIR=.\database"
SET "PYTHON_SCRIPT=run_baseframe.py"

REM # 定义 generator 和对应的 window_size 对
REM # Batch 没有直接的数组，我们用 FOR 循环和 token 解析来模拟配对
SET "generator_window_pairs=gru 5,lstm 10,transformer 15"

REM # 文件名到 start_timestamp 的映射 - 使用 IF 语句模拟
REM # 注意: Batch 文件名比较是大小写不敏感的
SET "SPECIAL_FILE_1=processed_原油_day.csv"
SET "START_TIME_1=1546"
SET "SPECIAL_FILE_2=processed_纸浆_day.csv"
SET "START_TIME_2=1710"

REM # 默认的 start_timestamp
SET "DEFAULT_START=31"

REM # 遍历数据文件
FOR %%F IN (%DATA_DIR%\processed_*_day.csv) DO (
    SET "FILENAME=%%~nxF"  REM # 获取文件名+扩展名 (e.g., processed_原油_day.csv)
    SET "BASENAME=%%~nF"   REM # 获取不带扩展名的文件名 (e.g., processed_原油_day)

    REM # 设置默认 start_timestamp
    SET "START_TIMESTAMP=%DEFAULT_START%"

    REM # 判断是否在特殊映射中以确定 START_TIMESTAMP
    IF /I "!FILENAME!"=="%SPECIAL_FILE_1%" (
        SET "START_TIMESTAMP=%START_TIME_1%"
    ) ELSE (
        IF /I "!FILENAME!"=="%SPECIAL_FILE_2%" (
            SET "START_TIMESTAMP=%START_TIME_2%"
        )
    )

    ECHO Processing data file: !FILENAME!
    ECHO -------------------------------------

    REM # 遍历 generator 和 window_size 组合 (使用逗号和空格作为分隔符)
    FOR %%P IN (%generator_window_pairs%) DO (
        REM # 解析出 generator 和 window_size
        FOR /F "tokens=1,2 delims=," %%A IN ("%%P") DO (
             SET "pair=%%A"
             FOR /F "tokens=1,2" %%G IN ("!pair!") DO (
                 SET "generator=%%G"
                 SET "window_size=%%H"

                 REM # === 定义特定于此组合的输出目录 ===
                 REM # 结构: .\output\<dataset_basename>\<generator_name>
                 SET "OUTPUT_DIR_COMBINED=.\output\baseframe\!BASENAME!\!generator!"

                 REM # 确保目录存在 (如果不存在则创建)
                 IF NOT EXIST "!OUTPUT_DIR_COMBINED!" (
                    MKDIR "!OUTPUT_DIR_COMBINED!"
                 )

                 ECHO Running with generator=!generator!, window_size=!window_size!, start=!START_TIMESTAMP!
                 ECHO Output directory: !OUTPUT_DIR_COMBINED!

                 REM # 执行 Python 脚本，传入所有参数
                 ECHO python "!PYTHON_SCRIPT!" --data_path "%%F" --output_dir "!OUTPUT_DIR_COMBINED!" --start_timestamp !START_TIMESTAMP! --generator !generator! --window_size !window_size!
                 python38 "!PYTHON_SCRIPT!" --data_path "%%F" --output_dir "!OUTPUT_DIR_COMBINED!" --start_timestamp !START_TIMESTAMP! --generator !generator! --window_size !window_size!


                 ECHO Finished run for generator=!generator!, window_size=!window_size!.
                 ECHO. REM # 输出空行
             )
        )
    )
    ECHO -------------------------------------
    ECHO Finished processing file: !FILENAME!
    ECHO.
)

ECHO All tasks completed.

ENDLOCAL
PAUSE REM # 可选：运行结束后暂停，方便查看输出