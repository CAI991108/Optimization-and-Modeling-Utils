# Optimization and Modeling Utils (Python)

This repository contains Python codes for various optimization and modeling tasks, 
including **inheritance partitioning**, **surface and contour plotting**, **warehouse location optimization**, 
**root-finding using Bisection and Golden Section methods**, **gradient descent with different step-size strategies**, 
**inertial gradient methods**, and **image inpainting**. 
Each script or notebook focuses on solving specific problems, such as minimizing functions, 
optimizing warehouse placement, or reconstructing damaged images, using techniques like brute-force search, 
gradient descent, and conjugate gradient methods. 
The repository also includes visualizations and performance comparisons, with dependencies on 
Python libraries like NumPy, SciPy, Matplotlib, and Jupyter Notebook. 

### Repository Structure
```
.
├── Inheritance/
│   ├── inheritance.py (.ipynb)
├── Surface and Contour Plot/
│   ├── plot.py (.ipynb)
├── Warehouse Location/
│   ├── warehouse.ipynb
├── Bisection and Golden Section/
│   ├── bi_golden_section.py (.ipynb)
├── (Inertial) Gradient Method/
│   ├── gradient_method.py 
│   ├── inertial_gradient.py
├── Smooth Image Inpainting/
│   ├── img_inpaint.ipynb
├── reports/
└── README.md
```


**Inheritance**
- The `inheritance.py` and the `inheritance.ipynb` aims to divide an inheritance list into three partitions, each intended for a daughter, 
such that the difference in the total value of each partition is minimized. 
The inheritance list contains weighted items, and the target value for each partition is one-third of the total inheritance. 
The script uses a brute-force approach to generate all possible combinations of partitions, calculates their sums, 
and finds the partition with the smallest maximum difference between the sums. 
If an optimal partition is found, it prints the three partitions along with their total values and the maximum difference. 
If no optimal partition is found, it outputs a message indicating so. 
The brute-force method is feasible due to the relatively small size of the inheritance list.

**Surface and Contour Plot**
- The `plot.py`  visualizes the function using `numpy` and `matplotlib`.
It creates a grid and evaluates the function over this grid, and generates two plots:
a 3D wireframe plot and a contour plot. 
The contour plot shows 50 levels with labeled contours and the same points marked. 
The script provides a clear visualization of the function's behavior in both 3D and 2D space.
- The `plot.ipynb` contains Python code for visualizing and optimizing a function. 
It generates a 3D surface plot of the function using `numpy` and `matplotlib`.
The script then uses `scipy.optimize.minimize` with the BFGS method to find a local minimum near the origin.
Additionally, it evaluates the function at four test points far from the origin, 
showing significant variations in function values. 
The code provides both visualization and optimization insights for the function.

**Warehouse Location**
- The `warehouse.ipynb` contains Python code for optimizing the placement of two warehouses 
using the `dual_annealing` algorithm from `scipy.optimize`. 
It defines separate objective functions for each warehouse, 
calculating costs based on distances to fixed points and additional fixed costs. 
The combined objective function minimizes the total cost for both warehouses. 
The variables are bounded within specific ranges, and the optimization finds the global minimum. 
The code efficiently solves the warehouse placement problem to minimize overall costs.

**Bisection and Golden Section**
- The `bi_golden_section.py` and the `bi_golden_section.ipynb` implements and compares 
the **Bisection Method** and **Golden Section Method** to find the minimum of the function.
The Bisection Method uses the derivative to narrow down the root, 
while the Golden Section Method directly minimizes using the golden ratio.
Both methods return the optimal value and the number of iterations required. 
The script also visualizes the function and the iteration paths of both methods using `matplotlib`, 
providing a clear comparison of their convergence behavior.

**(Inertial) Gradient Method**
- The `gradient_method.py` implements the **Gradient Descent Method** to minimize the function 
using three step-size strategies: **backtracking line search**, **diminishing step size**,
and **exact line search** (via the Golden Section Method). 
It starts from multiple initial points and iteratively updates the solution until convergence, 
tracking the number of iterations and the convergence path. 
The script also visualizes the function's contour plot and the convergence paths for each method, 
providing a clear comparison of their performance in finding the minimum. 
The results are printed for each initial point and method, showing the convergence points and iteration counts.
- The `inertial_gradient.py` implements the **Inertial Gradient Method** to minimize the function
using different values of the inertial parameter. 
The method iteratively updates the solution by combining gradient descent with an inertial term, 
ensuring convergence by satisfying the Armijo condition. The script starts from multiple initial points, 
tracks the function values, convergence paths, and iteration counts, and prints the results for each.
It also visualizes the function's contour plot and the convergence paths, 
providing insights into how different values affect the optimization process.

**Smooth Image Inpainting**
- This Python script performs **image inpainting** using a **Conjugate Gradient (CG) Method** 
and a **Standard Solver** (`spsolve`) to reconstruct damaged images based on a given mask. 
It constructs matrices `A` and `D` to represent the mask and gradient operators, respectively, 
and solves the system `Bx = C` with a regularization parameter `mu`. 
The script compares the performance of the CG method and `spsolve` in terms of **PSNR (Peak Signal-to-Noise Ratio)** 
and computation time, showing that the CG method is significantly faster while achieving the same PSNR.
It also visualizes the original and damaged images, demonstrating the inpainting process. 
The script is applied to two test images, showcasing its effectiveness in image reconstruction.


### Dependencies

- Python 3.x, NumPy, SciPy, Matplotlib, Scikit-learn, Jupyter Notebook

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
