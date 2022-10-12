import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

def f(x):
	return np.sin(10 * pi * x) / (2 * x) + (x - 1) ** 4

def df(x):
	return ((10 * pi * np.cos(10 * pi * x)) * (2 * x) - 2*(np.sin(10 * pi * x)))/4 * (x ** 2) +\
		(4*x**3 - 12*x**2 + 12*x - 4)

x = np.linspace(
	start = 0.5,
	stop = 2.5,
	num = 100
)

y = f(x)
plt.plot(x, y, color='#ddd')

beta = 2.4
alpha = 0.01

for i in range(1000):
	y = f(beta)
	plt.plot(beta, y, color=[0.9, 0.1, 0.4, 0.5], marker='o')
	if(i%10 == 0):
		alpha *= 0.98
	beta -= alpha * df(beta)

plt.show()