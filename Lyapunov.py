from typing import Union
import numpy as np
import pandas as pd
import hints 

class Lyapunov:
    def __init__(self,
                path: Union[str, None] = None,
                dt: Union[float, None] = None,
                alpha: Union[float, None] = None,
                A: Union[np.ndarray, None] = None,
                B: Union[np.ndarray, None] = None,
                C: Union[np.ndarray, None] = None,
        ):
        """
            explain the formula and the stochastic equation of that 
        """
        if A is None and B is None and C is None :
            raise ValueError("At least one of coefficient must be specified or get the path of dataset")
        elif dt is None:
            raise ValueError("the dt is needed")

        self.alpaha = alpha
        self.A = A
        self.B = B
        self.C = C
        self.path = path
        self.dt = dt
    
    def direct_method(self): 
        pass

    def Data_methods(self, order: int = 2):
        df_array = pd.read_csv(str(self.path)).to_numpy()
        calculator = hints.kmcc(ts_array=df_array, dt=self.dt, interaction_order=[ i for i in range(1, order+1)])



    
        
