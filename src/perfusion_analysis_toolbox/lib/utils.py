#!/usr/bin/env python3

import logging
import sys
import time


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def datestr():
    now = time.localtime()
    return "{:04}{:02}{:02}_{:02}{:02}".format(
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
    )
