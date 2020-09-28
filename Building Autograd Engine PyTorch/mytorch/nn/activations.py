import mytorch.nn.functional as F
from mytorch.nn.module import Module


class ReLU(Module):
    """ReLU activation function

    Example:
    >>> relu = ReLU()
    >>> relu(input_tensor)
    <some output>

    We run this class like a function because of Module.__call__().

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """ReLU forward pass.

        Args:
            x (Tensor): input before ReLU
        Returns:
            Tensor: input after ReLU
        """

        # Complete ReLU(Function) class in functional.py, then just call it here
        return x.ReLU()

# You can define more activation functions below (after hw1p1)
