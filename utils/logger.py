# logger_utils.py  ─────────────────────────────────────────────
import logging, json, os, datetime

def setup_experiment_logging(output_dir: str, args: dict,
                             log_name_prefix: str = "train") -> logging.Logger:
    """
    初始化实验日志。

    Parameters
    ----------
    output_dir : str
        训练脚本里传入的 output_dir，同 trainer 的可视化文件夹一致。
    args : dict
        所有需要记录的超参数 / CLI 解析结果，例如 vars(args)。
    log_name_prefix : str, optional
        生成文件名前缀，默认 "train"。

    Returns
    -------
    logging.Logger
        已经配置好的 logger，直接 logger.info(...) 使用。
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name_prefix}_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()          # 保留控制台输出
        ]
    )

    logger = logging.getLogger()          # root logger
    logger.info("ARGS = %s",
                json.dumps(args, ensure_ascii=False, indent=2))
    return logger
