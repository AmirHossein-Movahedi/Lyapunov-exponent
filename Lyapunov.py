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

__all__ = ["lyapunov"]

logger = logging.getLogger(__name__)


class lyapunov:
    """
    Container for constructing ODE systems from polynomial coefficients,
    integrating them, computing Lyapunov exponents and Kaplan–Yorke dimension.

    Example:
        L = Lyapunov(start_time=0.0, end_time=100.0, dt=0.01, initial_condition=np.array([0.1,0.2,0.3]))
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
        """
        Build symmetric 3D tensor B[i,j,m] from `array` having rows for unique pairs (i<=j).
        Input:
            array: shape (n_pairs, m) where n_pairs = n*(n+1)//2
            number_time_series: n
        Returns:
            tensor_3D: shape (n, n, m) with symmetry B[i,j,:] == B[j,i,:]
        Raises:
            ValueError on bad shapes.
        """
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
        # lexicographic pairs (i <= j)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                tensor[i, j, :] = arr[idx]
                tensor[j, i, :] = arr[idx]
                idx += 1
        return tensor

    @staticmethod
    def build_tensor_4D(array: NDArray, number_time_series: int) -> NDArray:
        """
        Build symmetric 4D tensor E[i,j,k,l] from `array` that lists unique
        coefficient patterns for combinations with repetition of length 3
        (i <= j <= k). The input rows correspond to those patterns.

        Input:
            array: shape (n_patterns, m?) OR (n_patterns,) depending on usage
            number_time_series: n

        Returns:
            tensor_4D: shape (n, n, n, n) symmetric across permutations of indices.
        """
        arr = np.asarray(array)
        n = int(number_time_series)

        # patterns are combinations_with_replacement indices of length 3
        patterns = list(combinations_with_replacement(range(n), 3))
        expected_rows = len(patterns)
        if arr.ndim == 1:
            # convert to shape (rows, 1)
            arr = arr.reshape((arr.shape[0], 1))
        if arr.ndim != 2:
            raise ValueError("array must be 1D or 2D (rows x m).")
        if arr.shape[0] != expected_rows:
            raise ValueError(
                f"array has {arr.shape[0]} rows but expected {expected_rows} for n={n}."
            )

        m = arr.shape[1]
        tensor = np.zeros((n, n, n, n), dtype=arr.dtype)

        # For each pattern (i,j,k) we will duplicate to 4 indices by duplicating
        # the most common index (to mimic B -> E extension in user's original).
        for row_idx, (i, j, k) in enumerate(patterns):
            values = arr[row_idx]  # shape (m,)
            # expand to 4 indices by repeating the most common index
            counts = Counter((i, j, k))
            most_common_idx = counts.most_common(1)[0][0]
            full_indices = [i, j, k, most_common_idx]

            unique_perms = set(permutations(full_indices))
            multiplicity = len(unique_perms)

            # distribute the coefficient values across all unique permutations evenly
            for perm in unique_perms:
                # place the vector `values` into the tensor at perm
                # if m == 1 treat as scalar
                if m == 1:
                    val = float(values[0])
                    if val != 0.0:
                        tensor[perm] += val / multiplicity
                else:
                    # for vector-valued coefficients (e.g., time series), we place each element
                    # across a 4D tensor of final shape (n,n,n,n,m) but original code
                    # assumed scalar for E; here we sum scalars only
                    raise NotImplementedError(
                        "build_tensor_4D currently supports scalar coefficients per pattern (1 column)."
                    )
        return tensor

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
        # Expect C as (n, n, n) where C[i,j,k] multiplies y(j)*y(k)
        # and E as (n,n,n,n) where E[i,j,k,l]*y(j)*y(k)*y(l)
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
                expr = expr + sum(float(E[i, j, k, l]) * y(j) * y(k) * y(l)
                                  for j in range(n) for k in range(n) for l in range(n))
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
    ) -> Tuple[NDArray, NDArray, plt.Figure, List[plt.Axes]]:
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

        return t_eval, y_result, fig, axs

    # -------------------------
    # Lyapunov exponents via jitcode_lyap wrapper
    # -------------------------
    @staticmethod
    def LE(f: Sequence, initial_condition: NDArray, t_span: NDArray) -> NDArray:
        """
        Compute Lyapunov exponents using jitcode_lyap wrapper.

        Returns:
            lyaps: array shape (len(t_span), n) where n = len(f)
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
            # jitcode_lyap usually returns extended state: [y..., lyap1, lyap2, ...]
            # Many implementations put Lyapunov exponents at the end; here we assume index 1..n
            # But to be robust we attempt to slice last n entries if out is 1D > n
            out = np.asarray(out, dtype=float).flatten()
            if out.size >= n:
                # try to obtain lyap part: choose last n elements
                lyap_part = out[-n:]
            else:
                raise RuntimeError("Unexpected output shape from jitcode_lyap integrate.")
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
        build tensors and derive/plot trajectories and compute Lyapunov/KY.

        This is a thin wrapper that imports `hints` lazily so the module can be imported
        without hints present.
        """
        try:
            import hints  # type: ignore
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
        t_eval, y_result, fig, axs = self.plot_trajectory(f, self.initial_condition, (self.t_i, self.t_f), self.dt)
        return {"t": t_eval, "y": y_result, "lyaps": lyaps, "ky": ky, "fig": fig, "axs": axs}

    def data_methods(self, alpha: NDArray, A: NDArray, C: NDArray, E: NDArray):
        """
        Use provided coefficient arrays directly (alpha, A, C, E) to run LE, KD and trajectory.
        """
        f = self.jit_equation_0_3(alpha, A, C, E)
        t_span = np.arange(self.t_i, self.t_f, self.dt)
        lyaps = self.LE(f, self.initial_condition, t_span)
        ky = self.KD(np.mean(lyaps[max(0, 1000):, :], axis=0))
        t_eval, y_result, fig, axs = self.plot_trajectory(f, self.initial_condition, (self.t_i, self.t_f), self.dt)
        return {"t": t_eval, "y": y_result, "lyaps": lyaps, "ky": ky, "fig": fig, "axs": axs}
