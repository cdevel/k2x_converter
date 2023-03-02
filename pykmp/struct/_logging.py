import logging

logger = logging.getLogger(__name__)


def setup_logging(name: str):
    global logger
    logger = logging.getLogger(name)

