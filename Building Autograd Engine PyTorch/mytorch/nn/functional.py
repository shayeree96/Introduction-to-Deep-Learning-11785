import numpy as np
import mytorch.tensor as tensor
from mytorch.autograd_engine import Function

def unbroadcast(grad,b):
    #print(" Shape of b in unbroadcast :",b.shape)
    #print("Shape of gradient in unbroadcast :",grad.shape)
    if b.shape:
       if grad.shape[0]==b.shape[0]:
            dim=1
       else:
            dim=0
       return grad.sum(axis=dim) 
    else:
        return 0

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        
        return tensor.Tensor(grad_output.data.T),None

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        
        requires_grad = a.requires_grad
        
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
   
        return c

    @staticmethod
    def backward(ctx, grad_output):
        
        grad=tensor.Tensor(grad_output.data.reshape(ctx.shape))
        #print("Grad in Reshape shape :",grad.shape)
        
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)),None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        
        #print("Log forward shape :",c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        
        
        a = ctx.saved_tensors[0]
        #print("Grad output Log Backward :",grad_output.shape)
        #print("a Log Backward :",a.shape)
        grad_a=tensor.Tensor(grad_output.data / a.data)
        #print("a in log:",a.shape)
        #print("grad_output in log:",grad_output.shape)
        
        #print("Shape of grad_a Log Backward :",grad_a.shape)
        if len(grad_output.shape)==1 and grad_output.shape!=grad_a.shape:
            grad_output.data=np.expand_dims(grad_output.data,axis=1)
        #print("Shape of grad_a Log Backward :",grad_output.shape)
        
        return tensor.Tensor(grad_output.data / a.data),None

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        #if a.data.shape != b.data.shape:
            #raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
       
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data
        
        if b.shape!=grad_b.shape:
            grad_b=unbroadcast(grad_b,b)
            
        if a.shape!=grad_a.shape:
            grad_a=unbroadcast(grad_a,a)     
        # the order of gradients returned should match the order of the arguments
        
        grad_a=tensor.Tensor(grad_a)
        grad_b=tensor.Tensor(grad_b) 
        # TODO: Implement more Functions below
        #print('grad_a and grad_b from Add :',grad_a.shape,grad_b.shape)
        return grad_a, grad_b


    
    
class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') :
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        #if a.data.shape!=b.data.shape:
            #raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))
        #raise Exception("TODO: Implement '-' forward")
        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)
        
        # Create subtraction output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data-b.data,requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        #print('Problem in Subtraction)
        
        return c
        
    @staticmethod
    def backward(ctx,grad_output):
        a, b = ctx.saved_tensors
        
        #raise Exception("TODO: Implement '-' backward")
        
        # calculate gradient of output w.r.t. each input
        
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = -1*np.ones(b.shape) * grad_output.data
        
        if len(b.shape)==1:
            grad_b=unbroadcast(grad_b,b)
            
        if len(a.shape)==1:
            grad_a=unbroadcast(grad_a,a)   
        # TODO: Implement more Functions below
        grad_a=tensor.Tensor(grad_a)
        grad_b=tensor.Tensor(grad_b) 
        # TODO: Implement more Functions below
        #print('grad_a and grad_b from Subtract :',grad_a.shape,grad_b.shape)
        return grad_a, grad_b

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') :
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        #if a.data.shape!=b.data.shape:
            #raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))
        #raise Exception("TODO: Implement '-' forward")
        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)
        
        # Create subtraction output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.multiply(a.data,b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        #raise Exception("TODO: Implement '-' backward")
        
        # calculate gradient of output w.r.t. each input
        grad_a = np.multiply(b.data,grad_output.data)
        grad_b = np.multiply(a.data,grad_output.data)
        
        if b.shape!=grad_b.shape:
            grad_b=unbroadcast(grad_b,b)
            
        if a.shape!=grad_a.shape:
            grad_a=unbroadcast(grad_a,a)  
        
        # TODO: Implement more Functions below
        grad_a=tensor.Tensor(grad_a)
        grad_b=tensor.Tensor(grad_b) 
        # TODO: Implement more Functions below
        #print('grad_a and grad_b from Mul :',grad_a.shape,grad_b.shape)
        return grad_a, grad_b
    
    
class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') :
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        if a.data.shape[1]!=b.data.shape[0]:
            raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))
        #raise Exception("TODO: Implement '-' forward")
        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)
        
        # Create subtraction output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.matmul(a.data,b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        #pdb.set_trace()
        #raise Exception("TODO: Implement '-' backward")
        # calculate gradient of output w.r.t. each input
        grad_a = np.matmul(grad_output.data,b.T().data)
        grad_b = np.matmul(a.T().data,grad_output.data)
        
        if b.shape!=grad_b.shape:
            grad_b=unbroadcast(grad_b,b)
            
        if a.shape!=grad_a.shape:
            grad_a=unbroadcast(grad_a,a)
        grad_a=tensor.Tensor(grad_a)
        grad_b=tensor.Tensor(grad_b) 
        # TODO: Implement more Functions below
        #print('grad_a and grad_b from Matmul :',grad_a.shape,grad_b.shape)
        return grad_a, grad_b
    
class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') :
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        #if a.data.shape!=b.data.shape:
            #raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))
        #raise Exception("TODO: Implement '-' forward")
        
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)
        
        # Create subtraction output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.divide(a.data,b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        
        return c
        
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        #raise Exception("TODO: Implement '-' backward")
        
        # calculate gradient of output w.r.t. each input
        
        grad_a = np.multiply(np.divide(1,b.data),grad_output.data)
        grad_b = -np.multiply(np.multiply(a.data,np.divide(1,b.data**2)),grad_output.data)
        
        if b.shape!=grad_b.shape:
            grad_b=tensor.Tensor(unbroadcast(grad_b,b))
            
        if a.shape!=grad_a.shape:
            grad_a=tensor.Tensor(unbroadcast(grad_a,a))   
        
        # TODO: Implement more Functions below
        #print('grad_a and grad_b from Division :',grad_a.shape,grad_b.shape)
        return grad_a,grad_b
    
def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
    
    a=tensor.Tensor((predicted.data).max())
    
    #print("Shape of a:",a.shape)
    diff=predicted-a
    #print("Diff shape:",diff.shape)
    exp=diff.Exp()
    #print("exp Diff shape:",exp.shape)
    sum_exp=exp.Sum_Column()
    #print("sum_exp shape:",sum_exp.shape)
    sum_exp_log=sum_exp.log()
    #print("sum_exp_log shape:",sum_exp_log.shape)
    
    add_with_a=sum_exp_log+a
    #print("add_with a shape:",add_with_a.shape)
    
    softmax=predicted-(((predicted-a).Exp().Sum_Column()).log()+a)
    #print("Shape of softmax :",softmax.shape)
    
    target=to_one_hot(target,num_classes)
    #print("Target shape :",target.shape)
    
    mul=(softmax * target)
    #print("Mul shape :",mul.shape)
    
    sum_mul_column=mul.Sum_Column()
    #print("Sum Mul Column shape :",sum_mul_column.shape)
    
    sum_row=sum_mul_column.Sum()
    #print("sum_row shape :",sum_row.shape)
    
    sum_over=(softmax * target).Sum_Column().Sum()
    #print("sum_over shape :",sum_over.shape)
    
    N=tensor.Tensor(batch_size)#Tensor for batch_size
    #print("N shape :",N.shape)
    
    NLLoss=sum_over/N#To ge the loss over all the batches
    
    #print("Shape of NLLoss :", NLLoss.shape)
    
    negate=tensor.Tensor(-1)
    
    #print("NLLoss :",NLLoss.shape)
    
    NLLoss=NLLoss.reshape()* negate
    
    return NLLoss
    

def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor') :
            raise Exception("Arg must be Tensors: {}, {}".format(type(a).__name__))
        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
    
        requires_grad = a.requires_grad 
        zero_check=np.zeros((1))
                
        a = tensor.Tensor(np.where(a.data<=zero_check,0,a.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return a
        
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors
        zero_check=np.zeros((1))
        grad_output = np.multiply(tensor.Tensor(np.where(a[0].data<=zero_check,0,1)),grad_output)
        #print("ReLU :",grad_output)
        # TODO: Implement more Functions below
        return grad_output, None
    
class Sum(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor') :
            raise Exception("Arg must be Tensors: {}, {}".format(type(a).__name__))
        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
    
        requires_grad = a.requires_grad 
        out = tensor.Tensor((a.data.sum(axis=0)), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        out=out.reshape(1,out.shape[0])
        #print("In sum Forward shape :",out.shape)
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        grad_a = tensor.Tensor(np.ones(a.shape)*grad_output.data)
        #print('grad_output shape in Sum_Row:',grad_a.shape)
      
        return grad_a, None
class Sum_Column(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor') :
            raise Exception("Arg must be Tensors: {}, {}".format(type(a).__name__))
        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
    
        requires_grad = a.requires_grad 
        out = tensor.Tensor((a.data.sum(axis=1)), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        
        
        out=out.reshape(out.shape[0],1)
        #print("In sum Column Forward shape :",out.shape)
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        
        
        a = ctx.saved_tensors[0]
        #print("Shape of a:",a.shape)
        
        #print("Shape of grad_ouput :",grad_output.shape)
        grad_a = tensor.Tensor(np.ones(a.shape)*grad_output.data)
        
        #print('grad_output shape in Sum_column:',grad_a.shape)
      
        return grad_a, None    
    
class Power(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor') :
            raise Exception("Arg must be Tensors: {}, {}".format(type(a).__name__))
        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
    
        requires_grad = a.requires_grad 
        
        #u=(x.sum(axis=0))/x.shape[0]         
        out = tensor.Tensor((a.data)**2, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        
        grad_output = tensor.Tensor(2*a.data*grad_output.data)
        #print("Shape of Power :",grad_output.shape)
        # TODO: Implement more Functions below
        #print("Power :",grad_output)
        return grad_output, None    
    
class Root(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor') :
            raise Exception("Arg must be Tensors: {}, {}".format(type(a).__name__))
        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
    
        requires_grad = a.requires_grad         
        out = tensor.Tensor(np.sqrt(a.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        
        grad_output = tensor.Tensor(0.5*((a.data)**(-0.5))*grad_output.data)
        #print("Root :",grad_output)
        #print("Shape of Root",grad_output.shape)
        # TODO: Implement more Functions below
        return grad_output, None     
    
class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor') :
            raise Exception("Arg must be Tensors: {}, {}".format(type(a).__name__))
        #raise Exception("TODO: Implement '-' forward")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)
    
        requires_grad = a.requires_grad         
        out = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return out
        
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        
        grad_output = tensor.Tensor(np.exp(a.data)*grad_output.data)
        #print("Root :",grad_output)
        #print("Shape of Root",grad_output.shape)
        # TODO: Implement more Functions below
        #print('grad_output shape in Exp:',grad_output.shape)
        return grad_output, None     
    