import time
import numpy as np

from torch4is.utils import time_log


def run():
    print(time_log())
    print("Hello World!")
    time.sleep(1)

    my_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    my_data2 = np.reshape(my_data, (2, 2))
    print(time_log())
    print(f"Numpy data:\n{my_data}, dtype: {my_data.dtype}, shape: {my_data.shape}")
    print(f"Numpy data2:\n{my_data2}, dtype: {my_data.dtype}, shape: {my_data2.shape}")
    print(f"Object info: {type(my_data)}, is numpy? {isinstance(my_data, np.ndarray)}")

if __name__ == '__main__':
    run()
