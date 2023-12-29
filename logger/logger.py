# -*- coding: utf-8 -*-

import logging


class Logger():
    """
    define logger
    """
    def __init__(self, logger_name='my_dog', logger_file='test.log'):
        self.logger_name = logger_name
        self.logger_file = logger_file
        # self.initialize(self.logger_name, self.logger_file)

    def get_logger(self):
        """
        config logger
        """
        # get logger
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        # set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # set file
        handler = logging.FileHandler(self.logger_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
