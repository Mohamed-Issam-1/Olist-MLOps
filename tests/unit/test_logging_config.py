import logging

import olist_ml.logging_config as logging_module


def reset_logger():
    logger = logging.getLogger(
        logging_module.LOGGER_NAME
    )

    for handler in list(
        logger.handlers
    ):
        handler.close()
        logger.removeHandler(
            handler
        )

    return logger


def test_configure_logging_creates_console_and_file_handlers(
    tmp_path,
    monkeypatch,
):
    reset_logger()

    monkeypatch.setattr(
        logging_module,
        "load_config",
        lambda: {
            "logging": {
                "level": "INFO",
                "directory": "logs",
                "application_log":
                    "application.log",
            }
        },
    )

    monkeypatch.setattr(
        logging_module,
        "resolve_project_path",
        lambda value:
            tmp_path / value,
    )

    logger = (
        logging_module
        .configure_logging()
    )

    handler_types = {
        type(handler)
        for handler in logger.handlers
    }

    assert (
        logging.StreamHandler
        in handler_types
    )

    assert (
        logging.FileHandler
        in handler_types
    )

    assert (
        tmp_path
        / "logs"
        / "application.log"
    ).exists()

    reset_logger()


def test_configure_logging_uses_configured_level(
    tmp_path,
    monkeypatch,
):
    reset_logger()

    monkeypatch.setattr(
        logging_module,
        "load_config",
        lambda: {
            "logging": {
                "level": "WARNING",
                "directory": "logs",
                "application_log":
                    "application.log",
            }
        },
    )

    monkeypatch.setattr(
        logging_module,
        "resolve_project_path",
        lambda value:
            tmp_path / value,
    )

    logger = (
        logging_module
        .configure_logging()
    )

    assert (
        logger.level
        == logging.WARNING
    )

    reset_logger()


def test_configure_logging_does_not_duplicate_handlers(
    tmp_path,
    monkeypatch,
):
    reset_logger()

    monkeypatch.setattr(
        logging_module,
        "load_config",
        lambda: {
            "logging": {
                "level": "INFO",
                "directory": "logs",
                "application_log":
                    "application.log",
            }
        },
    )

    monkeypatch.setattr(
        logging_module,
        "resolve_project_path",
        lambda value:
            tmp_path / value,
    )

    first = (
        logging_module
        .configure_logging()
    )

    first_count = len(
        first.handlers
    )

    second = (
        logging_module
        .configure_logging()
    )

    second_count = len(
        second.handlers
    )

    assert first is second
    assert first_count == 2
    assert second_count == 2

    reset_logger()


def test_get_logger_returns_child_logger(
    tmp_path,
    monkeypatch,
):
    reset_logger()

    monkeypatch.setattr(
        logging_module,
        "load_config",
        lambda: {
            "logging": {
                "level": "INFO",
                "directory": "logs",
                "application_log":
                    "application.log",
            }
        },
    )

    monkeypatch.setattr(
        logging_module,
        "resolve_project_path",
        lambda value:
            tmp_path / value,
    )

    logger = (
        logging_module
        .get_logger(
            "inference"
        )
    )

    assert (
        logger.name
        == "olist_ml.inference"
    )

    reset_logger()
