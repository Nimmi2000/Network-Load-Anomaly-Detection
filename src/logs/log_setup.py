import logging
import os
from datetime import datetime

def get_logger(name:str):
    os.makedirs('logs', exist_ok=True)

    log_filename = datetime.now().strftime('logs/%Y-%m-%d_%H-%M-%S.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler() 
        ]
    )

    return logging.getLogger(name)