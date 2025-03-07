import logging
import os
import sys
from datetime import datetime


def mk_and_get_out_dir():
    t_start = datetime.now()
    dir = f"output/logs/job_name-{t_start.strftime('%y_%m_%d-%H_%M_%S')}"
    os.makedirs(dir, exist_ok=True)
    return dir


def config_logging(out_dir=None):

    log_to_file = False
    console_level = 20
    file_level = 10 if log_to_file else console_level

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s: %(lineno)d >> %(message)s')

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if log_to_file:
        if out_dir is None:
            out_dir = mk_and_get_out_dir()
        _logging_file = f"{out_dir}/logging.log"
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
