from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module
import pdb

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        #print("x.shape[0]",x.shape[0])  
        if self.is_train:
            u=x.Sum()/Tensor(x.shape[0])
            #print("In forward shape of u:",u.shape)
            s=(((x-u).Power()).Sum())/Tensor(x.shape[0])
            #print("In forward shape of s:",s.shape)
            x_new=(x-u)/(s+self.eps).Root()
            #print("In forward shape of x_new:",x_new.shape)
            y=(self.gamma*x_new) + self.beta
            
            var=((x-u).Power().Sum())/Tensor(x.shape[0]-1)
            self.running_mean=(Tensor(1)-self.momentum)*self.running_mean+(self.momentum*u)
            self.running_var=(Tensor(1)-self.momentum)*self.running_var+(self.momentum*var)

            return y
        else:
            u=self.running_mean
            #print("In forward shape of u:",u.shape)
            s=self.running_var
            #print("In forward shape of s:",s.shape)
            x_new=(x-u)/(s+self.eps).Root()
            #print("In forward shape of x_new:",x_new.shape)
            y=(self.gamma*x_new) + self.beta
            
            return y
            
            
        
    
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
