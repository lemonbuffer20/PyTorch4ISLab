import time
import numpy as np

from torch4is.utils import time_log


def run():
    print(time_log())
    print("Hello World!")
    time.sleep(1)

    data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    data2 = np.reshape(data, (2, 2))
    print(time_log())
    print(f"Numpy data:\n{data}, dtype: {data.dtype}, shape: {data.shape}")
    print(f"Numpy data2:\n{data2}, dtype: {data.dtype}, shape: {data2.shape}")
    print(f"Object info: {type(data)}, is numpy? {isinstance(data, np.ndarray)}")




if __name__ == '__main__':
    run()
