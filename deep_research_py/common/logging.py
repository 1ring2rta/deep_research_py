import logging
import datetime
import os


logger = None


def initial_logger(logging_path: str = "log", enable_stdout: bool = False) -> None:
    """Initializes the logger for the application."""
    global logger

    now = datetime.datetime.now()
    log_file = os.path.join(
        logging_path, f"/mnt/data/hanchen/deep-research-py/logs/deep_research_py_{now.strftime('%Y%m%d_%H%M%S')}.log"
    )
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    # Set up logging to stdout if enabled
    handlers = [logging.FileHandler(log_file)]
    if enable_stdout:
        handlers.append(logging.StreamHandler())
    # Set up logging to a file
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=handlers
    )
    logger = logging.getLogger("deep_research_py")
    logger.setLevel(logging.INFO)


def log_event(event_desc: str) -> None:
    """Logs an event with input and output token counts."""
    if not logger:
        return
    logger.info(f"Event: {event_desc}")


def log_error(error_desc: str) -> None:
    """Logs an error message."""
    if not logger:
        return
    logger.error(f"Error: {error_desc}")


def log_warning(warning_desc: str) -> None:
    """Logs a warning message."""
    if not logger:
        return
    logger.warning(f"Warning: {warning_desc}")
