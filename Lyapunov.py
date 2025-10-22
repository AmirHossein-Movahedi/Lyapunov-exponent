from typing import Union
from collections import Counter
from itertools import permutations
import numpy as np
import pandas as pd
import hints 
from jitcsde import jitcsde, y, t
from jitcode import jitcode_lyap, y, jitcode
import matplotlib.pyplot as plt
from scipy.stats import sem
from numpy.typing import NDArray


class Lyapunov:

    def __init__(self,
                start_time:Union[float, None] = 0.001,
                end_time:Union[float, None]= None,
                dt:Union[float, None] = None,
                initial_condition:Union[NDArray, None] = None,
        ):
        """
            the necessary values for Lyapunov
        """
        if end_time is None:
            raise ValueError("Need the end time")
        elif dt is None:
            raise ValueError("the dt is needed")
        elif initial_condition is None:
            raise ValueError("order needed")

        self.t_i = start_time
        self.t_f = end_time
        self.dt = dt
        self.initial_condition = initial_condition
    
    
    @staticmethod
    def build_tensor_3D(array:NDArray, number_time_series:int):
        array = np.asarray(array)
        num_pairs_expected = number_time_series * (number_time_series + 1) // 2

        if array.ndim != 2:
            raise ValueError("rows_array must be 2D with shape (num_pairs, m).")
        if array.shape[0] != num_pairs_expected:
            raise ValueError(f"rows_array has {array.shape[0]} rows but expected {num_pairs_expected} for n={number_time_series}.")
        
        m = array.shape[1]
        # generate pairs in lexicographic order (i <= j)
        pairs = [(i, j) for i in range(number_time_series) for j in range(i, number_time_series)]

        tensor_3D = np.zeros((number_time_series, number_time_series, m), dtype=array.dtype)

        for (i, j), values in zip(pairs, array):
            tensor_3D[i, j, :] = values
            tensor_3D[j, i, :] = values

        return tensor_3D
    

    @staticmethod
    def build_tensor_4D(array:NDArray, number_time_series:int):
        array = np.asarray(array)
        tensor_4D = np.zeros((number_time_series, number_time_series, number_time_series, number_time_series))
        
        # generate all index patterns of length 3 with non-decreasing order
        # e.g., for n=4 → (0,0,0), (0,0,1), ..., (3,3,3)
        patterns = []
        for i in range(number_time_series):
            for j in range(i, number_time_series):
                for k in range(j, number_time_series):
                    patterns.append((i, j, k))
        
        # Now each row of matrix_array corresponds to one of these patterns
        for row_idx, indices in enumerate(patterns):
            values = array[row_idx]
            
            # Build full 4 indices by duplicating one index
            # (so pattern length 3 → tensor rank 4)
            counter = Counter(indices)
            full_indices = []
            for idx, count in counter.items():
                full_indices.extend([idx] * count)
            
            if len(full_indices) == 3:  # pad to 4
                most_common = counter.most_common(1)[0][0]
                full_indices.append(most_common)
            
            # Generate all unique permutations
            unique_perms = set(permutations(full_indices))
            multiplicity = len(unique_perms)
            
            for val in values:
                if val != 0:
                    for perm in unique_perms:
                        tensor_4D[perm] += val / multiplicity
        
        return tensor_4D
    

    @staticmethod
    def jit_equation_0_3(alpha:NDArray, A:NDArray, C:NDArray, E:NDArray):
        """
        Constructs the derivatives ẋ_i(t) for a system, including:
        Const[i]        : constant offsets
        A[i,j]*y(j)     : linear terms
        B[i,j,k]*y(j)*y(k)  : quadratic terms
        E[i,j,k,l]*y(j)*y(k)*y(l) : cubic terms

        Parameters
        ----------
        Const : array_like
            1D array of shape (n,), the constant offsets.
        A : array_like
            2D array of shape (n, n), the linear coefficients.
        B : array_like
            3D array of shape (n, n, n), the quadratic coefficients.
        E : array_like
            4D array of shape (n, n, n, n), the cubic coefficients.
    
        Returns
        -------
        eq : list
            A list of symbolic expressions suitable for JIT compilation with jitcsde.
        """
        # Convert all inputs to numpy asarray
        alpha = np.asarray(alpha)
        A = np.asarray(A)
        C = np.asarray(C)
        E = np.asarray(E)
        
        n = A.shape[0]
        eq = []
        for i in range(n):
            # Build up the i-th equation
            expression = (
                alpha                                                   # constant term
                + sum(A[i, j] * y(j) for j in range(n))                 # Linear term
                # Quadratic term, summing only for j <= k to avoid duplicates
                + sum(C[i, j, k] * y(j) * y(k) 
                    for j in range(n) 
                    for k in range(j, n))
                # Cubic term, summing only for j <= k <= l
                + sum(E[i, j, k, l] * y(j) * y(k) * y(l)
                    for j in range(n) 
                    for k in range(j, n) 
                    for l in range(k, n))
            )
            eq.append(expression)
        return eq
    

    ################################### PLOT ########################################
    @staticmethod
    def plot_trajectory(f, initial_condition, t_span, dt):
        # Initialize the JIT-compiled ODE system
        ode = jitcode(f)
        ode.set_integrator("dopri5")
        ode.set_initial_value(initial_condition, t_span[0])

        # Create time evaluation points
        t_eval = np.arange(t_span[0], t_span[1], dt)
        y_result = np.empty((len(t_eval), len(initial_condition)))

        # Integrate the system
        for i, t in enumerate(t_eval):
            y_result[i] = ode.integrate(t)

        # Plot the trajectory
        if len(initial_condition) == 2:
            plt.plot(y_result[:, 0], y_result[:, 1], lw=0.5)
            plt.xlabel('y0')
            plt.ylabel('y1')
            plt.title('Phase Space Trajectory')
            plt.figure(figsize=(15,5))
            plt.plot(t_eval, y_result[:, 0], label='y0')
            plt.plot(t_eval, y_result[:, 1], label='y1')
        elif len(initial_condition) == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(y_result[:, 0], y_result[:, 1], y_result[:, 2], lw=0.5)
            ax.set_xlabel('y0')
            ax.set_ylabel('y1')
            ax.set_zlabel('y2')
            ax.set_title('3D Phase Space Trajectory')
            plt.figure(figsize=(15, 5))
            plt.plot(t_eval, y_result[:, 0], label='y0')
            plt.plot(t_eval, y_result[:, 1], label='y1')
            plt.plot(t_eval, y_result[:, 2], label='y2')
        elif len(initial_condition) > 3:
            for i in range(len(initial_condition)):
                plt.figure(figsize=(10, 3))
                plt.plot(t_eval, y_result[:, i], label=f'y{i}')
        return y_result


    @staticmethod
    def LE(f,initial_condition,t_span):
        n = len(f)
        ODE = jitcode_lyap(f, n_lyap=n)
        ODE.set_integrator("dopri5")
        ODE.set_initial_value(initial_condition,0.0)

        lyaps = []
        for time in t_span:
            lyaps.append(ODE.integrate(time)[1])

        # converting to NumPy array for easier handling
        lyaps = np.vstack(lyaps)

        for i in range(n):
            lyap = np.average(lyaps[1000:,i])
            stderr = sem(lyaps[1000:,i])
            print("%i. Lyapunov exponent: % .4f ± %.4f" % (i+1,lyap,stderr))
        return lyaps
    
    
    @staticmethod
    def KD(lyaps):
        lyaps=np.round(lyaps,2)
        s = 0
        j=0
        sorted_lyaps = np.sort(lyaps)[::-1]
        if np.all(sorted_lyaps > 0):
            print('repelling(diverging in all directions)')
            return
        elif np.all(sorted_lyaps < 0):
            print('attractor')
            return print('D=',0)
        else:
            for i in range(len(sorted_lyaps)):
                s_old = np.copy(s)
                s += sorted_lyaps[i]
                if s>=0:j+=1
                else:break
                D = j + np.sum(sorted_lyaps[:j])/np.abs(sorted_lyaps[j])
                return D.tolist()


    def direct_method(self, path:str, order:int, build_tensor_3D, build_tensor_4D, jit_equation_0_3, plot_trajectory, LE, KD):
        #load data
        df_array = pd.read_csv(path).to_numpy()
        hints_calculator = hints.kmcc(ts_array=df_array, dt=self.dt, interaction_order=[i for i in range(0, order+1)])
        coefficient = hints_calculator.get_coefficients()
        number_time_series = len(df_array[0])

        alpha = np.asarray(coefficient.iloc[0,:])
        A = np.asarray(coefficient.iloc[:number_time_series, :])
        C = np.asarray(coefficient.iloc[number_time_series:int((number_time_series*(number_time_series+1))/2)+number_time_series, :])
        E = np.asarray(coefficient.iloc[int((number_time_series*(number_time_series+1))/2)+number_time_series:, :])

        C = build_tensor_3D(C, number_time_series)
        E = build_tensor_4D(E, number_time_series)

        f = jit_equation_0_3(alpha, A, C, E)
        t_span = np.arange(self.t_i, self.t_f, self.dt) 
        lyaps_np_th=LE(f,self.initial_condition,t_span)
        KD(np.mean(lyaps_np_th[1000:,:],axis=0))
        ts_np_th = plot_trajectory(f, initial_condition=self.initial_condition, t_span=(self.t_i,self.t_f), dt=self.dt)


    def Data_methods(self, alpha:NDArray, A:NDArray, C:NDArray,  E:NDArray, build_tensor_3D, build_tensor_4D, jit_equation_0_3,
                     plot_trajectory, LE, KD):
        
        # Make sure your inputs are properly shaped numpy arrays
        f = jit_equation_0_3(alpha, A, C, E)
        t_span = np.arange(self.t_i, self.t_f, self.dt)
        lyaps_np_th=LE(f,self.initial_condition,t_span)
        KD(np.mean(lyaps_np_th[1000:,:],axis=0))
        ts_np_th = plot_trajectory(f, initial_condition=self.initial_condition, t_span=(self.t_i,self.t_f), dt=self.dt)
        




    
        
