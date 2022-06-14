import random as rnd
import warnings
from typing import List

import numpy as np


class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


def fsign(f: float) -> float:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


def abs_max(n: int, a: List[float]) -> float:
    """np.max(np.abs(a))"""
    m: float = abs(a[0])

    for i in range(1, n):
        d: float = abs(a[i])
        if d > m:
            m = d

    return m


def biclustering_lasso_coordinate_descent(
    w,
    alpha,
    X,
    max_iter,
    tol
):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    upper_left = np.zeros((n_samples, n_samples))
    lower_right = np.zeros((n_features, n_features))
    left = np.concatenate((upper_left, X.T))
    right = np.concatenate((X, lower_right))
    jordan_wielandt = 0.5 * np.concatenate((left, right), axis=1)

    gap = 0  # TODO: duality gap

    n_iter = 0
    for n_iter in range(max_iter):
        w_max: float = 0.0
        d_w_max: float = 0.0
        for ii in range(n_features + n_samples):
            w_ii: float = w[ii]  # Store previous value

            # tmp is regression approximation
            tmp = np.dot(w, jordan_wielandt[ii]) - jordan_wielandt[ii][ii] * w[ii]

            if tmp >= 0 and abs(tmp) > alpha:
                w[ii] = (-alpha - tmp) / jordan_wielandt[ii][ii]
            elif tmp >= 0 and abs(tmp) <= alpha:
                w[ii] = 0
            elif tmp < 0 and abs(tmp) > alpha:
                w[ii] = (alpha - tmp) / jordan_wielandt[ii][ii]
            else:
                w[ii] = 0

            # update the maximum absolute coefficient update
            d_w_ii: float = abs(w[ii] - w_ii)
            d_w_max = max(d_w_max, d_w_ii)

            w_max = max(w_max, abs(w[ii]))

        # check, if the iterations converged and we can break here
        if d_w_max < tol:
            break

            # TODO: calculate the duality gap
    else:
        # for/else, runs if for doesn't end with a `break`
        message = (
            "Objective did not converge. You might want to increase "
            "the number of iterations, check the scale of the "
            "features or consider increasing regularisation. "
            f"Duality gap: {gap:.3e}, tolerance: {tol:.3e}"
        )
        if alpha < np.finfo(np.float64).eps:
            message += (
                " Linear regression models with null weight for the "
                "l1 regularization term are more efficiently fitted "
                "using one of the solvers implemented in "
                "sklearn.linear_model.Ridge/RidgeCV instead."
            )
        warnings.warn(message, ConvergenceWarning)

    return w, gap, tol, n_iter + 1


def lasso_coordinate_descent(
    w,
    alpha,
    X,
    y,
    max_iter,
    tol
):
    """Coordinate descent algorithm for Lasso regression.

        We minimize

        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1)
    """
    dtype = np.float32

    # get the data information into easy vars
    n_samples: int = X.shape[0]
    n_features: int = X.shape[1]

    # compute norms of the columns of X
    norm_cols_X = np.square(X).sum(axis=0)

    # initial value of the residuals
    R = np.empty(n_samples, dtype=dtype)
    XtA = np.empty(n_features, dtype=dtype)

    gap: float = tol + 1.0
    d_w_tol: float = tol

    if alpha == 0:
        warnings.warn("Coordinate descent with no regularization may lead to "
                      "unexpected results and is discouraged.")

    R = y - np.dot(X, w)

    tol *= np.dot(y, y)

    for n_iter in range(max_iter):
        w_max: float = 0.0
        d_w_max: float = 0.0
        for ii in range(n_features):  # Loop over coordinates
            if norm_cols_X[ii] == 0.0:
                continue

            w_ii: float = w[ii]  # Store previous value

            if w_ii != 0.0:
                R += w_ii * X[:,ii]

            tmp: float = (X[:,ii]*R).sum()

            w[ii] = (fsign(tmp) * max(abs(tmp) - alpha, 0) / norm_cols_X[ii])  # / (norm_cols_X[ii] + beta))

            if w[ii] != 0.0:
                R -= w[ii] * X[:,ii]  # Update residual

            # update the maximum absolute coefficient update
            d_w_ii: float = abs(w[ii] - w_ii)
            d_w_max = max(d_w_max, d_w_ii)

            w_max = max(w_max, abs(w[ii]))

        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller
            # than the tolerance: check the duality gap as ultimate
            # stopping criterion

            XtA = np.dot(X.T, R)  # - beta * w
            dual_norm_XtA: float = np.max(np.abs(XtA))

            R_norm2: float = np.dot(R, R)

            if (dual_norm_XtA > alpha):
                const: float = alpha / dual_norm_XtA
                A_norm2: float = R_norm2 * (const ** 2)
                gap: float = 0.5 * (R_norm2 + A_norm2)
            else:
                const: float = 1.0
                gap = R_norm2

            l1_norm: float = np.abs(w).sum()

            gap += (alpha * l1_norm - const * np.dot(R.T, y))

            if gap < tol:
                # return if we reached desired tolerance
                break

    else:
        # for/else, runs if for doesn't end with a `break`
        message = (
            "Objective did not converge. You might want to increase "
            "the number of iterations, check the scale of the "
            "features or consider increasing regularisation. "
            f"Duality gap: {gap:.3e}, tolerance: {tol:.3e}"
        )
        if alpha < np.finfo(np.float64).eps:
            message += (
                " Linear regression models with null weight for the "
                "l1 regularization term are more efficiently fitted "
                "using one of the solvers implemented in "
                "sklearn.linear_model.Ridge/RidgeCV instead."
            )
        warnings.warn(message, ConvergenceWarning)

    return w, gap, tol, n_iter + 1
