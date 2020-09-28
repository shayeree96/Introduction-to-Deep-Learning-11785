import numpy as np

import mytorch.autograd_engine as autograd_engine
from mytorch.nn import functional as F
from mytorch.autograd_engine import AccumulateGrad


class Tensor:
    """Tensor object, similar to `torch.Tensor`
    A wrapper around a NumPy array that help it interact with MyTorch.

    Args:
        data (np.array): the actual data of the tensor
        requires_grad (boolean): If true, accumulate gradient in `.grad`
        is_leaf (boolean): If true, this is a leaf tensor; see writeup.
        is_parameter (boolean): If true, data contains trainable params
    """
    def __init__(self, data, requires_grad=False, is_leaf=True,
                 is_parameter=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = None # Set during forward pass
        self.grad = None
        self.is_parameter = is_parameter

    # ------------------------------------
    # [Not important] For printing tensors
    # ------------------------------------
    def __str__(self):
        return "{}{}".format(
            str(self.data),
            ", grad_fn={}".format(self.grad_fn.__class__.__name__) if self.grad_fn is not None else ""
        )

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------
    # Tensor Operations (NOT part of comp graph)
    # ------------------------------------------
    @property
    def shape(self):
        """Returns the shape of the data array in a tuple.
        >>> a = Tensor(np.array([3,2])).shape
        (2,)
        """
        return self.data.shape

    def fill_(self, fill_value):
        """In-place operation, replaces data with repeated value"""
        self.data.fill(fill_value)
        return self

    def copy(self):
        """Returns copy of this tensor
        Note: after copying, you may need to set params like `is_parameter` manually"""
        return Tensor(self.data)

    # Below methods can be used WITHOUT creating a tensor first
    # (For example, we can call Tensor.zeros(3,2) directly)

    @staticmethod
    def zeros(*shape):
        """Creates new tensor filled with 0's
        Args:
            shape: comma separated ints i.e. Tensor.zeros(3,4,5)
        Returns:
            Tensor: filled w/ 0's
        """
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(*shape):
        """Creates new tensor filled with 1's
        Note: if you look up "asterik args python", you'll see this function is
        called as follows: ones(1, 2, 3), not: ones((1, 2, 3))
        """
        return Tensor(np.ones(shape))

    @staticmethod
    def arange(*stop):
        """Creates new tensor filled by `np.arange()`"""
        return Tensor(np.arange(stop))

    @staticmethod
    def randn(*shape):
        """Creates new tensor filled by normal distribution (mu=0, sigma=1)"""
        return Tensor(np.random.normal(0, 1, shape))

    @staticmethod
    def empty(*shape):
        """Creates an tensor with uninitialized data (NOT with 0's).

        >>> Tensor.empty(1,)
        [6.95058141e-310]
        """
        return Tensor(np.empty(shape))

    # ----------------------
    # Autograd backward init
    # ----------------------
    def backward(self):
        """Kicks off autograd backward (see writeup for hints)"""
        #raise Exception("TODO: Kick off `autograd_engine.backward()``")
        #here we kick off the backward to autograd engine.backward()
        return autograd_engine.backward(self.grad_fn, Tensor(np.ones(self.shape)))
    # ------------------------------------------
    # Tensor Operations (ARE part of comp graph)
    # ------------------------------------------
    def T(self):
        
        """Transposes data (for 2d data ONLY)

        >>> Tensor(np.array([[1,2,3],[4,5,6]])).T()
        [[1, 4],
         [2, 5],
         [3, 6]]
        """
        return F.Transpose.apply(self)

    def reshape(self, *shape):
        """Makes new tensor of input shape, containing same data
        (NOT in-place operation)

        >>> Tensor(np.array([[1,2,3],[4,5,6]])).reshape(3,2)
        [[1, 2],
         [3, 4],
         [5, 6]]
        """
        return F.Reshape.apply(self, shape)

    def log(self):
        """Element-wise log of this tensor, adding to comp graph"""
        return F.Log.apply(self)

    def __add__(self, other):
        """Links "+" to the comp. graph
        Args:
            other (Tensor): other tensor to add
        Returns:
            Tensor: result after adding
        """
        return F.Add.apply(self, other)

    def __sub__(self, other):
        #raise Exception("TODO: Link '-' to comp. graph, like in __add__()")       
        return F.Sub.apply(self, other)
        
    def __mul__(self, other):
        #raise Exception("TODO: Link '*' to comp. graph, like in __add__()")
        
        return F.Mul.apply(self, other)
    
    def __truediv__(self, other):
        #raise Exception("TODO: Link '/' to comp. graph, like in __add__()")
        
        return F.Div.apply(self, other)
    def __matmul__(self,other):
        
        return F.MatMul.apply(self,other)
    # TODO: Implement more functions below
    
    def ReLU(self):
        
        return F.ReLU.apply(self)
    
    def Sum(self):
        
        return F.Sum.apply(self)
    
    def Sum_Column(self):
        
        return F.Sum_Column.apply(self)
    
    def Power(self):
        
        return F.Power.apply(self)
    
    def Root(self):
        
        return F.Root.apply(self)
    
    def Exp(self):
        
        return F.Exp.apply(self)