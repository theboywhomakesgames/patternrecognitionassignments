"""
Calculate mean vector of each class in ds1.
Plot each mean vector as seq i.e. the horizontal axis should indicate the component of the vector from 0-1023. The vertical axis should indicate the mean in that component.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

paths = glob.glob("./1/ds/training_validation/*")
data = []

for filePath in paths:
	x = []
	with open(filePath) as f:
		for i in range(32):
			line = f.readline();
			for c in line:
				if(c == '0' or c == '1'):
					x.append(0 if c == '0' else 1)
	data.append(x)

data = np.array(data)

means = np.zeros((1024,))
for sample in data:
	for idx, component in enumerate(sample):
		means[idx] += component

means /= 1024
indices = [i for i in range(1024)]
plt.bar(indices, means)
plt.show()