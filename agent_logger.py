import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("agent")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.propagate = False

if not logger.handlers:
    # 控制台日志：便于在 VS Code 终端查看
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件日志：单个文件最多 5 MB，保留 3 个备份
    file_handler = RotatingFileHandler(
        LOG_DIR / "agent.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
