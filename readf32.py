import numpy as np


def read_float32_from_binary(first_col: int, last_col: int, path: str,
                             num_rows: int) -> np.array:
    """Read set of columns from column major matrix stored in a binary file

    The binary file is expected to contain a flattened, column major matrix
    of 32bit floats.

    Args:
        first_col (int): (0-based) index of first column to read
        last_col (int): (0-based) index of last column to read
        path (str): path to binary file
        num_rows (int): number of rows per column in the binary file

    Returns:
        np.array: num_rows x num_read_columns numpy array with dtype float32 values
    """
    bytes_per_col = 4 * num_rows
    offset = bytes_per_col * first_col
    num_cols_to_read = last_col - first_col + 1
    num_floats_to_read = num_cols_to_read * num_rows
    return np.fromfile(path,
                       dtype=np.float32,
                       count=num_floats_to_read,
                       offset=offset).reshape(num_rows,
                                              num_cols_to_read,
                                              order="F")


def write_float32_to_binary(arr: np.array, path: str):
    """Write a linear array of float32 values to a binary file.

    Args:
        arr (np.array): array of values to be writte to file
        path (str): path to output file
    """
    arr.astype(np.float32).tofile(path)