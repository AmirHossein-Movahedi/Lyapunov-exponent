from Lyapunov1 import Lyapunov
import numpy as np
import pandas as pd 

initial_conditions = np.random.rand(3)

file = np.array(pd.read_csv("UPO.csv").iloc[:,1:])
df_array = np.array(file)


L = Lyapunov(start_time=0.0001, end_time=60, dt=0.001,initial_condition=initial_conditions)
result = L.output_hints_method(file)
print(result["t"])
print("\n -------------------------- \n")
print(result["y"])
print("\n -------------------------- \n")
print(result["lyaps"])
print("\n -------------------------- \n")
print(result["ky"])
print("\n -------------------------- \n")
