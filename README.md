# Lyapunov-exponent
Calculate the Lyapunov exponent for a stochastic differential equation characterized by the coefficients 

# Installation
```bash
git clone 
cd lyapunov
pip install -e .
```
# Usage

Below is a basic example 

```python
from Lyapunov1 import Lyapunov
import numpy as np

L = Lyapunov(start_time=0.0, end_time=50.0, dt=0.01, initial_condition=np.array([0.1, 0.2, 0.3]))

alpha = np.zeros(3)
A = np.eye(3)
C = np.zeros((3, 3, 3))
E = np.zeros((3, 3, 3, 3))

res = L.data_methods(alpha, A, C, E)
print(res["ky"])
```
