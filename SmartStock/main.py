from concurrent.futures import ThreadPoolExecutor, as_completed

import tushare as ts
import pandas as pd
import time
import os
import logging
import numpy as np

from SmartStock.LinearRegression import LinearRegression
from SmartStock.MathTools import getAbsMaxIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    logger = logging.getLogger("main")

    start_time = time.time()
    file_list = os.listdir("/home/faquanyao/work_space/stocks/data/")
    models = []
    for data_file in file_list:
        if os.path.splitext(data_file)[1] == '.csv':
            per_start_time = time.time()
            logger.info("Process data " + str(data_file))
            df = pd.read_csv("/home/faquanyao/work_space/stocks/data/"+data_file)
            y = df['close']
            x = np.arange(0, len(y))
            model = LinearRegression(x, y)
            models.append({data_file : model})
            per_end_time = time.time()
            logger.info("Process data {} cost {} seconds.".format(data_file, per_end_time - per_start_time))

            y_pre = model.predict(x)

            getAbsMaxIndex(y, y_pre)



    end_time = time.time()
    logger.info("Process all data cost {} seconds.".format(end_time - start_time))

    ###########show data ###############
    logger.info("show data")
    start_time = time.time()
    for data_file in file_list:
        if os.path.splitext(data_file)[1] == '.csv':
            pass


