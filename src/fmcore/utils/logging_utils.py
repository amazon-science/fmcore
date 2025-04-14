import logging


class Log:
    """
    A logging class that provides direct access to logging methods.
    Uses Python’s built-in logging module, which is already thread-safe.
    """

    _logger = logging.getLogger("sysout_logger")
    _logger.setLevel(logging.INFO)

    # Ensure the logger is configured only once
    if not _logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    @classmethod
    def debug(cls, msg, *args, **kwargs):
        cls._logger.debug(msg, *args, **kwargs)

    @classmethod
    def info(cls, msg, *args, **kwargs):
        cls._logger.info(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg, *args, **kwargs):
        cls._logger.warning(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg, *args, **kwargs):
        cls._logger.error(msg, *args, **kwargs)

    @classmethod
    def critical(cls, msg, *args, **kwargs):
        cls._logger.critical(msg, *args, **kwargs)

    @classmethod
    def get_logger(cls):
        return cls._logger
