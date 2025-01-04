import matplotlib.pyplot as plt
import numpy as np
from mpmath import sqrt


# The objective function
def f(x):
    x1, x2 = x
    return 0.5 * x1 ** 4 - x1 ** 3 - x1 ** 2 + x1 ** 2 * x2 ** 2 + 0.5 * x2 ** 4 - x2 ** 2


# The gradient of the objective function
def grad_f(x):
    x1, x2 = x
    df_dx1 = 2 * x1 ** 3 - 3 * x1 ** 2 - 2 * x1 + 2 * x1 * x2 ** 2
    df_dx2 = 2 * x1 ** 2 * x2 + 2 * x2 ** 3 - 2 * x2
    return np.array([df_dx1, df_dx2])


# Backtracking line search
def backtracking_line_search(obj, grad, x, alpha, sigma, gamma):
    while obj(x - alpha * grad(x)) > obj(x) - gamma * alpha * np.dot(grad(x).T, grad(x)):
        alpha *= sigma
    return alpha


# Diminishing step size
def diminishing_step_size(k):
    return 1 / np.sqrt(k + 2)


# Golden section method for exact line search
def golden_section_search(obj, x, grad, tol, maxit):
    a = 0
    b = 1
    phi = (1 + np.sqrt(5)) / 2

    for _ in range(maxit):
        d = (phi - 1) * b + (2 - phi) * a
        c = (2 - phi) * b + (phi - 1) * a

        if obj(x - c * grad(x)) < obj(x - d * grad(x)):
            b = d
        else:
            a = c

        if np.abs(b - a) < tol:
            break

    return (a + b) / 2


# The gradient descent method (aggregate of all the above methods)
def gradient_method(obj, grad, x0, tol, maxit=None, method='backtracking'):
    x = x0
    strategy = [x0]

    for k in range(maxit):
        if method == 'backtracking':
            alpha = backtracking_line_search(obj, grad, x, alpha=1, sigma=0.5, gamma=0.1)
        elif method == 'diminishing':
            alpha = diminishing_step_size(k+1)
        elif method == 'exact':
            alpha = golden_section_search(obj, x, grad, tol=1e-6, maxit=100)
        else:
            raise ValueError('Invalid method')

        x_new = x - alpha * grad(x)
        strategy.append(x_new)

        if np.linalg.norm(grad(x)) < tol:
            break

        x = x_new

    return x, k, strategy


# The set of initial points - X0
X0 = np.array([
    [-0.5, 1],
    [-0.5, 0.5],
    [-0.25, -0.5],
    [0.5, -0.5],
    [0.5, 1]
])

# Run the gradient descent method for each initial point
results_backtracking = []
results_diminishing = []
results_exact = []


def gd_backtracking():
    for x0 in X0:
        x_star, k, strategy = gradient_method(f, grad_f, x0, tol=1e-5, maxit=10000, method='backtracking')
        results_backtracking.append((x0, k, x_star))

    for i, (x0, k, x_star) in enumerate(results_backtracking):
        print(f'Backtracking: initial point: {x0}, iterations: {k}, convergence point: {x_star}')


def gd_diminishing():
    for x0 in X0:
        x_star, k, strategy = gradient_method(f, grad_f, x0, tol=1e-5, maxit=10000, method='diminishing')
        results_diminishing.append((x0, k, x_star))

    for i, (x0, k, x_star) in enumerate(results_diminishing):
        print(f'Diminishing: initial point: {x0}, iterations: {k}, convergence point: {x_star}')


def gd_exact():
    for x0 in X0:
        x_star, k, strategy = gradient_method(f, grad_f, x0, tol=1e-6, maxit=100, method='exact')
        results_exact.append((x0, k, x_star))

    for i, (x0, k, x_star) in enumerate(results_exact):
        print(f'Exact: initial point: {x0}, iterations: {k}, convergence point: {x_star}')


if __name__ == "__main__":
    gd_backtracking()
    gd_diminishing()
    gd_exact()

    # Define contour plot range for the function
    x1 = np.linspace(-2.5, 2.5, 300)
    x2 = np.linspace(-2.5, 2.5, 300)
    X1, X2 = np.meshgrid(x1, x2)
    function = f([X1, X2])

    # Define the different step-size methods to compare
    methods = ["backtracking", "diminishing", "exact"]

    # Generate plots for each method from each initial point
    for method in methods:
        plt.figure(figsize=(10, 8))

        custom_levels = np.concatenate((np.linspace(-10, 5, 30), [6, 7, 8, 9, 10]))

        contour = plt.contour(X1, X2, function, levels=custom_levels, cmap="viridis")
        plt.colorbar(contour)

        for x0 in X0:
            _, _, path = gradient_method(f, grad_f, x0, tol=1e-5, maxit=10000, method=method)
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], marker='o', label=f"initial point: {x0}")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Convergence with Gradient Method: {method}")
        plt.legend()
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.show()
