import logging


class Logger:
    def __init__(self, logger_name, log_file):
        # create logger object
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # create file handler and set level to DEBUG
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # add formatter to file handler
        fh.setFormatter(formatter)

        # add file handler to logger
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger
