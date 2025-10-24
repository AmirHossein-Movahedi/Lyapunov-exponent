"""
Lyapunov module suitable for packaging.

Primary class: Lyapunov
Functions:
 - build_tensor_3D, build_tensor_4D : build symmetric coefficient tensors
 - jit_equation_0_3 : compose RHS expressions for jitcode / jitcsde
 - plot_trajectory : integrate and return trajectories + Matplotlib figures
 - LE : compute Lyapunov exponents via jitcode_lyap (wrap)
 - KD : Kaplan–Yorke dimension calculator

"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, List
from collections import Counter
from itertools import permutations, combinations_with_replacement
import logging

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import sem
import matplotlib.pyplot as plt

"""
Notes:
 - This module uses `jitcsde`, `jitcode` and `jitcode_lyap`. If those
   packages are not installed the module raises an informative ImportError.
"""


# Conditional imports for optional heavy dependencies
try:
    from jitcsde import jitcsde, y, t  # optional; needed for DDE-style systems
except Exception:
    jitcsde = None
    y = None
    t = None

try:
    from jitcode import jitcode_lyap, y as _y_jc, jitcode
    # prefer jitcode's `y` if jitcsde isn't installed
    if y is None:
        y = _y_jc
except Exception:
    jitcode_lyap = None
    jitcode = None

# Attempt to import user-provided helper module 'hints' at runtime in methods that need it.
# Do not import here to allow package tests when hints isn't present.

__all__ = ["Lyapunov"]

logger = logging.getLogger(__name__)


class Lyapunov:
    """
    Container for constructing ODE systems from polynomial coefficients,
    integrating them, computing Lyapunov exponents.

    Example:
        L = Lyapunov(start_time=0.0001, end_time=100.0, dt=0.01, initial_condition=np.array([0.1,0.2,0.3]))
    """

    def __init__(
        self,
        start_time: Optional[float] = 0.001,
        end_time: Optional[float] = None,
        dt: Optional[float] = None,
        initial_condition: Optional[NDArray] = None,
    ) -> None:
        if end_time is None:
            raise ValueError("end_time must be provided.")
        if dt is None:
            raise ValueError("dt must be provided.")
        if initial_condition is None:
            raise ValueError("initial_condition must be provided.")

        self.t_i = float(start_time or 0.0)
        self.t_f = float(end_time)
        self.dt = float(dt)
        self.initial_condition = np.asarray(initial_condition, dtype=float).copy()

    # -------------------------
    # Tensor builders
    # -------------------------
    @staticmethod
    def build_tensor_3D(array: NDArray, number_time_series: int) -> NDArray:
        arr = np.asarray(array)
        n = int(number_time_series)
        expected_rows = n * (n + 1) // 2
        if arr.ndim != 2:
            raise ValueError("array must be 2D with shape (n_pairs, m).")
        if arr.shape[0] != expected_rows:
            raise ValueError(
                f"array has {arr.shape[0]} rows but expected {expected_rows} for n={n}."
            )

        m = arr.shape[1]
        tensor = np.zeros((n, n, m), dtype=arr.dtype)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                tensor[i, j, :] = arr[idx]
                tensor[j, i, :] = arr[idx]
                idx += 1
        return tensor


    @staticmethod
    def build_tensor_4D(array: NDArray, number_time_series: int) -> NDArray:
        arr = np.asarray(array)
        n = int(number_time_series)
        patterns = list(combinations_with_replacement(range(n), 3))
        expected_rows = len(patterns)

        if arr.ndim == 1:
            arr = arr.reshape((arr.shape[0], 1))
        if arr.shape[0] != expected_rows:
            raise ValueError(f"array has {arr.shape[0]} rows but expected {expected_rows} for n={n}.")

        m = arr.shape[1]
        tensor = np.zeros((m, n, n, n, n), dtype=arr.dtype)  # one 4D tensor per column

        for row_idx, (i, j, k) in enumerate(patterns):
            counts = Counter((i, j, k))
            most_common_idx = counts.most_common(1)[0][0]
            full_indices = [i, j, k, most_common_idx]
            unique_perms = set(permutations(full_indices))
            multiplicity = len(unique_perms)

            for col in range(m):
                val = float(arr[row_idx, col])
                if val != 0.0:
                    for perm in unique_perms:
                        tensor[col, perm[0], perm[1], perm[2], perm[3]] += val / multiplicity

        return tensor  # shape: (m, n, n, n, n)


    # -------------------------
    # Construct JIT expressions
    # -------------------------
    @staticmethod
    def jit_equation_0_3(
        alpha: NDArray, A: NDArray, C: NDArray, E: NDArray
    ) -> List:
        """
        Create list of expressions for JIT integrators (jitcode / jitcsde).

        alpha: 1D shape (n,) constants per equation
        A: 2D shape (n, n) linear terms
        C: 3D shape (n, n, ?) expects C[i,j,k] where k maps to state index OR
           if C is shape (n,n,n) treat as quadratic coefficients over state indices.
        E: 4D shape (n,n,n,n)

        Returns:
            list of expressions (one per equation) using `y(j)` references for jitcode.
        """
        alpha = np.asarray(alpha)
        A = np.asarray(A)
        C = np.asarray(C)
        E = np.asarray(E)

        n = int(A.shape[0])
        if alpha.shape[0] != n:
            raise ValueError("alpha must have length n (A.shape[0])")

        eqs = []
        for i in range(n):
            # constant per-equation
            expr = float(alpha[i])
            # linear terms
            expr = expr + sum(float(A[i, j]) * y(j) for j in range(n))
            # quadratic terms
            if C.ndim == 3 and C.shape[0] == n:
                expr = expr + sum(float(C[i, j, k]) * y(j) * y(k) for j in range(n) for k in range(n))
            elif C.ndim == 4 and C.shape[0] == n:
                # if provided full 4D, fallback (rare)
                expr = expr + sum(float(C[i, j, k, l]) * y(j) * y(k) * y(l)
                                  for j in range(n) for k in range(n) for l in range(n))
            else:
                # assume C provided in the form (n,n) zero-case
                pass

            # cubic terms from E
            if E is not None and E.size != 0:
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            val = E[i, j, k, l]
                            if np.ndim(val) > 0:  # if it's a small array
                                val = val.item() if val.size == 1 else float(np.mean(val))
                            expr += float(val) * y(j) * y(k) * y(l)

            eqs.append(expr)
        return eqs

    # -------------------------
    # Plotting / trajectory
    # -------------------------
    @staticmethod
    def plot_trajectory(
        f: Sequence,
        initial_condition: NDArray,
        t_span: Tuple[float, float],
        dt: float,
    ):
        """
        Integrate `f` (list of jitcode expressions) with jitcode and produce trajectories.

        Returns:
            t_eval, y_result, fig, axs
        """
        if jitcode is None:
            raise RuntimeError("jitcode is required for plot_trajectory (install jitcode).")

        ode = jitcode(f)
        ode.set_integrator("dopri5")
        ode.set_initial_value(np.asarray(initial_condition, dtype=float), float(t_span[0]))

        t_eval = np.arange(t_span[0], t_span[1], dt, dtype=float)
        y_result = np.empty((len(t_eval), len(initial_condition)), dtype=float)

        for idx, tt in enumerate(t_eval):
            y_result[idx] = ode.integrate(tt)

        # build plot(s)
        fig = plt.figure(constrained_layout=True, figsize=(10, 5))
        axs = []

        dim = len(initial_condition)
        if dim == 2:
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(y_result[:, 0], y_result[:, 1], lw=0.7)
            ax.set_xlabel("y0")
            ax.set_ylabel("y1")
            ax.set_title("Phase Space")
            axs.append(ax)

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(t_eval, y_result[:, 0], label="y0")
            ax2.plot(t_eval, y_result[:, 1], label="y1")
            ax2.legend()
            axs.append(ax2)
        elif dim == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.plot(y_result[:, 0], y_result[:, 1], y_result[:, 2], lw=0.7)
            ax.set_xlabel("y0")
            ax.set_ylabel("y1")
            ax.set_zlabel("y2")
            axs.append(ax)

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(t_eval, y_result[:, 0], label="y0")
            ax2.plot(t_eval, y_result[:, 1], label="y1")
            ax2.plot(t_eval, y_result[:, 2], label="y2")
            ax2.legend()
            axs.append(ax2)
        else:
            ax = fig.add_subplot(1, 1, 1)
            for i in range(dim):
                ax.plot(t_eval, y_result[:, i], label=f"y{i}")
            ax.legend()
            axs.append(ax)

        fig.savefig("./out.png")

        return t_eval, y_result

    # -------------------------
    # Lyapunov exponents via jitcode_lyap wrapper
    # -------------------------
    @staticmethod
    def LE(f: Sequence, initial_condition: NDArray, t_span: NDArray) -> NDArray:
        """
        Compute Lyapunov exponents using jitcode_lyap wrapper.
        Returns:
            lyaps: array shape (len(t_span), n)
        """
        if jitcode_lyap is None:
            raise RuntimeError("jitcode_lyap is required for LE (install jitcode).")

        n = len(f)
        ODE = jitcode_lyap(f, n_lyap=n)
        ODE.set_integrator("dopri5")
        ODE.set_initial_value(np.asarray(initial_condition, dtype=float), float(t_span[0]))

        lyaps_list = []

        for tt in t_span:
            out = ODE.integrate(tt)

            # handle tuple or nested structure
            if isinstance(out, (tuple, list)):
                out = np.concatenate([np.ravel(np.asarray(o, dtype=float)) for o in out])
            else:
                out = np.ravel(np.asarray(out, dtype=float))

            # sanity check
            if out.size < n:
                raise RuntimeError(f"Unexpected output shape {out.shape} from jitcode_lyap integrate.")

            lyap_part = out[-n:]
            lyaps_list.append(lyap_part)

        return np.vstack(lyaps_list)


    # -------------------------
    # Kaplan–Yorke / KY dimension
    # -------------------------
    @staticmethod
    def KD(lyaps: NDArray) -> Optional[float]:
        """
        Compute Kaplan–Yorke dimension from a 1D array of Lyapunov exponents (sorted descending).

        Returns:
            D: float or None if undefined.
        """
        arr = np.asarray(lyaps, dtype=float).flatten()
        # sort descending
        arr_sorted = np.sort(arr)[::-1]

        # If all positive or all negative handle trivial cases
        if np.all(arr_sorted > 0):
            logger.info("All Lyapunov exponents positive: repeller.")
            return None
        if np.all(arr_sorted < 0):
            logger.info("All Lyapunov exponents negative: attractor (D=0).")
            return 0.0

        s = 0.0
        j = 0
        for idx, lam in enumerate(arr_sorted):
            s += lam
            if s >= 0:
                j = idx + 1
            else:
                # cumulative sum crossed negative at this lam, compute D using previous partial sum
                if idx == 0:
                    return 0.0
                sum_pos = np.sum(arr_sorted[:j])
                if j < len(arr_sorted):
                    lam_next = arr_sorted[j]
                    D = j + sum_pos / abs(lam_next) if lam_next != 0 else float(j)
                    return float(D)
                return float(j)
        # if loop completes, all cumulative sums >= 0
        return float(j)

    # -------------------------
    # High-level convenience methods
    # -------------------------
    def direct_method(self, path: str, order: int):
        """
        Read CSV time-series at `path`, compute coefficients using hints.kmcc,
        build tensors and derive/plot trajectories and compute Lyapunov.

        This is a thin wrapper that imports `hints` lazily so the module can be imported
        without hints present.
        """
        try:
            import hints  
        except Exception as exc:
            raise RuntimeError(
                "direct_method requires the external `hints` module (hints.kmcc)."
            ) from exc

        df_array = pd.read_csv(path).to_numpy()
        hints_calc = hints.kmcc(ts_array=df_array, dt=self.dt, interaction_order=[i for i in range(0, order + 1)])
        coefficient = hints_calc.get_coefficients()

        number_time_series = df_array.shape[1]
        # slicing coefficients: adapted from original but clarified
        alpha = np.asarray(coefficient.iloc[0, :])
        # linear block
        A = np.asarray(coefficient.iloc[1:number_time_series + 1, :])
        # quadratic rows: next n*(n+1)//2 rows
        q_start = 1 + number_time_series
        q_end = q_start + (number_time_series * (number_time_series + 1)) // 2
        C_rows = np.asarray(coefficient.iloc[q_start:q_end, :])
        e_rows = np.asarray(coefficient.iloc[q_end:, :])

        C = self.build_tensor_3D(C_rows, number_time_series)
        E = self.build_tensor_4D(e_rows, number_time_series)

        f = self.jit_equation_0_3(alpha, A, C, E)
        t_span = np.arange(self.t_i, self.t_f, self.dt)
        lyaps = self.LE(f, self.initial_condition, t_span)
        ky = self.KD(np.mean(lyaps[max(0, 1000):, :], axis=0))
        t_eval, y_result = self.plot_trajectory(f, self.initial_condition, (self.t_i, self.t_f), self.dt)
        for i in range(number_time_series):
            lyap = np.average(lyaps[1000:,i])
            stderr = sem(lyaps[1000:,i])
            print("%i. Lyapunov exponent: % .4f ± %.4f" % (i+1,lyap,stderr))

        return {"t": t_eval, "y": y_result, "lyaps": lyaps, "ky": ky}

    def data_methods(self, alpha: NDArray, A: NDArray, C: NDArray, E: NDArray):
        """
        Use provided coefficient arrays directly (alpha, A, C, E) to run LE, KD and trajectory.
        """
        f = self.jit_equation_0_3(alpha, A, C, E)
        t_span = np.arange(self.t_i, self.t_f, self.dt)
        lyaps = self.LE(f, self.initial_condition, t_span)
        ky = self.KD(np.mean(lyaps[max(0, 1000):, :], axis=0))
        t_eval, y_result = self.plot_trajectory(f, self.initial_condition, (self.t_i, self.t_f), self.dt)
        for i in range(len(self.initial_condition)):
            lyap = np.average(lyaps[1000:,i])
            stderr = sem(lyaps[1000:,i])
            print("%i. Lyapunov exponent: % .4f ± %.4f" % (i+1,lyap,stderr))
        return {"t": t_eval, "y": y_result, "lyaps": lyaps, "ky": ky}

    def output_hints_method(self, df_array: NDArray):

        number_time_series = df_array.shape[1]
        coefficient = pd.DataFrame(df_array)
        alpha = np.asarray(coefficient.iloc[0, :])
        # linear block
        A = np.asarray(coefficient.iloc[1:number_time_series + 1, :])
        # quadratic rows: next n*(n+1)//2 rows
        q_start = 1 + number_time_series
        q_end = q_start + (number_time_series * (number_time_series + 1)) // 2
        C_rows = np.asarray(coefficient.iloc[q_start:q_end, :])
        e_rows = np.asarray(coefficient.iloc[q_end:, :])
        C = self.build_tensor_3D(C_rows, number_time_series)
        E = self.build_tensor_4D(e_rows, number_time_series)
        f = self.jit_equation_0_3(alpha, A, C, E)
        t_span = np.arange(self.t_i, self.t_f, self.dt)
        lyaps = self.LE(f, self.initial_condition, t_span)
        ky = self.KD(np.mean(lyaps[max(0, 1000):, :], axis=0))
        t_eval, y_result, = self.plot_trajectory(f, self.initial_condition, (self.t_i, self.t_f), self.dt)
        for i in range(number_time_series):
            lyap = np.average(lyaps[1000:,i])
            stderr = sem(lyaps[1000:,i])
            print("%i. Lyapunov exponent: % .4f ± %.4f" % (i+1,lyap,stderr))
        return {"t": t_eval, "y": y_result, "lyaps": lyaps, "ky": ky}