import numpy as np
import matplotlib.pyplot as plt

mu = np.array([3, 7])
s = np.array([
	[6.9, -4.9],
	[-4.9, 7]
])

points = np.random.multivariate_normal(mu, s, (5000,))
# plt.scatter(points[:, 0], points[:, 1], s=0.1)
# plt.show()

cov = np.cov(points.T)
print("the new cov matrix:")
print(cov)

U, s, V = np.linalg.svd(points)
S = np.zeros((2, 2))
S[:2, :2] = np.diag(s)

new_data = np.array([V @ S @ p for p in points])
plt.scatter(new_data[:, 0], new_data[:, 1], s=0.1)
plt.show()