import torch
import torch.nn.functional as F

def pad_one_to_last(nums, total_length, axis=-1, val=0):
    """Pad val to last in nums in given axis.

    length of the result in given axis should be total_length.

    Raises:
      IndexError: If the input axis value is out of range of the nums array

    Args:
        nums (numpy.ndarray): The array to pad.
        total_length (int): The final width of the Array.
        axis (int): Axis along which a sum is performed.
        val (int): The value to set the padded value.

    Returns:
        torch.Tensor: Padded array

    """
    tensor = torch.Tensor(nums)
    axis = (axis + len(tensor.shape)) if axis < 0 else axis

    if len(tensor.shape) <= axis:
        raise IndexError('axis {} is out of range {}'.format(
            axis, tensor.shape))

    padding_config = [0, 0] * len(tensor.shape)
    padding_idx = abs(axis - len(tensor.shape)) * 2 - 1
    padding_config[padding_idx] = max(total_length - tensor.shape[axis], val)
    return F.pad(tensor, padding_config, value=1)