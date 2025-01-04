import numpy as np
from A4_1 import f, grad_f, X0
import matplotlib.pyplot as plt


# The inertial gradient method function
def inertial_gradient_method(obj, grad, x0, tol, beta, L):
    xk = x0
    xk0 = x0
    alpha = 1.99 * (1 - beta) / L

    function_values = [obj(xk)]
    strategy = [xk.copy()]
    iteration = 0

    while True:
        # Check for convergence
        if np.linalg.norm(grad(xk)) <= tol:
            return xk, function_values, np.array(strategy), iteration

        # Compute the update
        xk1 = xk - alpha * grad(xk) + beta * (xk - xk0)
        iteration += 1

        # Check if the Armijo condition is satisfied
        while obj(xk1) - obj(xk) >= np.dot(grad(xk), xk1 - xk) + (L / 2) * np.linalg.norm(xk1 - xk) ** 2:
            L *= 2
            alpha = 1.99 * (1 - beta) / L
            xk1 = xk - alpha * grad(xk) + beta * (xk - xk0)

        # Update the current point
        xk0 = xk
        xk = xk1
        function_values.append(obj(xk))
        strategy.append(xk.copy())


if __name__ == '__main__':
    # Define the different values of beta
    betas = [0.3, 0.5, 0.7, 0.9]
    tol = 1e-5

    # Define contour plot range for the function
    x1 = np.linspace(-2.5, 2.5, 300)
    x2 = np.linspace(-2.5, 2.5, 300)
    X1, X2 = np.meshgrid(x1, x2)
    function = f([X1, X2])

    # Generate plots for each method from each initial point
    for beta in betas:
        plt.figure(figsize=(10, 8))

        custom_levels = np.concatenate((np.linspace(-10, 5, 30), [6, 7, 8, 9, 10]))

        contour = plt.contour(X1, X2, function, levels=custom_levels, cmap="viridis")
        plt.colorbar(contour)

        print(f'Inertial Gradient Method with beta {beta}: ')

        for x0 in X0:
            x_star, function_values, path, iteration = inertial_gradient_method(f, grad_f, x0, tol, beta, L=beta)
            print(f'initial point: {x0}, iterations: {iteration}, convergence point: {x_star}')

            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], marker='o', label=f"initial point: {x0}")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Convergence with Inertial Gradient Method (beta = {beta})")
        plt.legend()
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.show()