from __future__ import annotations

import logging

from olist_ml.config import (
    load_config,
    resolve_project_path,
)


LOGGER_NAME = "olist_ml"


def configure_logging() -> logging.Logger:
    """
    Configure application logging once.

    Logs are written to both:
    - console
    - application log file
    """

    config = load_config()
    logging_config = config["logging"]

    logger = logging.getLogger(
        LOGGER_NAME
    )

    if logger.handlers:
        return logger

    level_name = str(
        logging_config.get(
            "level",
            "INFO",
        )
    ).upper()

    level = getattr(
        logging,
        level_name,
        None,
    )

    if not isinstance(
        level,
        int,
    ):
        raise ValueError(
            f"Invalid logging level: {level_name}"
        )

    log_directory = resolve_project_path(
        logging_config[
            "directory"
        ]
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    application_log = (
        log_directory
        / logging_config[
            "application_log"
        ]
    )

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setLevel(
        level
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler = logging.FileHandler(
        application_log,
        encoding="utf-8",
    )

    file_handler.setLevel(
        level
    )

    file_handler.setFormatter(
        formatter
    )

    logger.setLevel(
        level
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    logger.propagate = False

    return logger


def get_logger(
    name: str | None = None,
) -> logging.Logger:
    """
    Return a child logger under the main olist_ml logger.
    """

    configure_logging()

    if not name:
        return logging.getLogger(
            LOGGER_NAME
        )

    return logging.getLogger(
        f"{LOGGER_NAME}.{name}"
    )
