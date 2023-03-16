import os.path
import numpy as np
from readf32 import read_float32_from_binary, write_float32_to_binary

TEST_DATA_NC = 4
TEST_DATA_NR = 3
TEST_DATA = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32)
TEST_DATA_PATH = "./test_data.bin"


def test_read_write_float32():
    """Test reading and writing from/to binary of linearized float32 matrices
    """
    try:
        os.remove(TEST_DATA_PATH)
    except FileNotFoundError:
        pass
    write_float32_to_binary(TEST_DATA, TEST_DATA_PATH)
    assert os.path.isfile(TEST_DATA_PATH)
    obs = read_float32_from_binary(1, 2, TEST_DATA_PATH, TEST_DATA_NR)
    exp = TEST_DATA.reshape(TEST_DATA_NR, TEST_DATA_NC, order="F")[:, 1:3]
    assert (obs == exp).all()
    os.remove(TEST_DATA_PATH)