import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 定义函数
def f3(x1, x2):
    return x1 ** 4 + 2 * (x1 - x2) * x1 ** 2 + 4 * x2 ** 2


# 创建x1和x2的网格
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)

# 计算函数值
Z = f3(X1, X2)

# 创建图形
fig = plt.figure(figsize=(14, 6))

# 绘制3D线框图（类似于等高线）
ax1 = fig.add_subplot(211, projection='3d')
ax1.plot_wireframe(X1, X2, Z, color='k', linewidth=0.5)
ax1.set_title('Surface Plot of $f_3(x)$')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('$f_3(x)$')
ax1.scatter(0, 0, f3(0, 0), color='r', label='(0, 0)')
ax1.scatter(-2, 1, f3(-2, 1), color='b', label='(-2, 1)')
ax1.legend()

# 绘制等高线图
ax2 = fig.add_subplot(212)
contour = ax2.contour(X1, X2, Z, levels=50, colors='k')
ax2.set_title('Contour Plot of $f_3(x)$')
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.scatter(0, 0, color='r')
ax2.scatter(-2, 1, color='b')
ax2.clabel(contour, inline=True, fontsize=10)  # 标注等高线的值

plt.show()
