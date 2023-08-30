import numpy as np


def average_reshape(array, new_shape):
    """
    Averages the array along the last dimension, to give it a new shape.

    Parameters:
    - array: ndarray
        The input array to be averaged.
    - new_shape: tuple
        The desired shape of the output array. The last dimension should be smaller
        than the corresponding dimension of the input array.

    Returns:
    - ndarray: the averaged array with the new shape.
    """
    # Calculate the factor by which to reduce the last dimension
    reduction_factor = array.shape[-1] // new_shape[-1]

    # Validate
    if array.shape[-1] % new_shape[-1] != 0:
        raise ValueError(
            "The last dimension of the new shape must evenly divide the last dimension of the original array.")

    # Validate rest of the dimensions
    for dim_old, dim_new in zip(array.shape[:-1], new_shape[:-1]):
        if dim_old != dim_new:
            raise ValueError("All dimensions except the last should remain unchanged.")

    # Reshape and average
    reshaped_array = np.mean(array.reshape(*new_shape[:-1], new_shape[-1], reduction_factor), axis=-1)

    return reshaped_array


# Create a random array with shape (1000, 3, 2000)
array = np.random.rand(1000, 3, 2000)

# Target new shape (1000, 3, 200)
new_shape = (1000, 3, 200)

# Average and reshape
averaged_array = average_reshape(array, new_shape)

# Verify shape
print(averaged_array.shape)