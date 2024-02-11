from pathlib import Path
import logging.config

def setup_logger(
        logger_name: str,
        log_level: str ='INFO',
        log_file: str or None =None) -> logging.Logger:
    """
    Example
    -------
    >>> import logger_config
    >>> logger = logger_config.setup_logger('my_logger', log_level='DEBUG')
    >>> logger.info("my log")
    """
    config = {
        'version': 1,
        'formatters': {
            'default': {
                'format': '%(asctime)s|%(levelname)s| %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': log_level
            },
        },
        'root': {
            'handlers': ['console'],
            'level': log_level
        }
    }
    if log_file:
        log_file_ = Path(log_file)
        assert log_file_.parent.exists()
        assert not log_file_.is_dir(), "log_file must be a file path"
        config['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'formatter': 'default',
                'level': log_level,
                'filename': log_file_
            }
        config['root']['handlers'].append('file')
    logging.config.dictConfig(config)

    return logging.getLogger(logger_name)
