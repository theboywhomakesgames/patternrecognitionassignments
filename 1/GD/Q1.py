import numpy as np
import matplotlib.pyplot as plt

def f(x):
	return x**2

def df(x):
	return 2*x

x = np.linspace(
	start = -5,
	stop = 5,
	num = 100
)

y = f(x)
plt.plot(x, y, color='#ddd')

beta = 4.5
alpha = 0.01

for i in range(1000):
	y = f(beta)
	plt.plot(beta, y, color='orange', marker='o')
	beta -= alpha * df(beta)

plt.show()