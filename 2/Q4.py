import numpy as np
import glob
import re

# read the data
paths = glob.glob("./1/ds/training_validation/*")
data = []
labels = []

for filePath in paths:
	x = re.findall("(\d+)", filePath);
	labels.append(int(x[-2]))

	x = []
	with open(filePath) as f:
		for i in range(32):
			line = f.readline();
			x.append(int(line, base=2))
	data.append(x)

data = np.array(data)
data = data / data.std()

# seperate the data by label
classes = []
cur_label = ''
iter = 0
tmp = []
for label, sample in zip(labels, data):
	if cur_label != label:
		cur_label = label
		if(iter > 0):
			classes.append(tmp)
			tmp = []

	tmp.append(sample)
	iter += 1

classes.append(tmp)

Z = np.dot(data.T, data)
eigenvalues, eigenvectors = np.linalg.eig(Z)

D = np.diag(eigenvalues)
P = eigenvectors

Z_new = np.dot(Z, P)

#1. Calculate the proportion of variance explained by each feature
sum_eigenvalues = np.sum(eigenvalues)
prop_var = [i/sum_eigenvalues for i in eigenvalues]

#2. Calculate the cumulative variance
cum_var = [np.sum(prop_var[:i+1]) for i in range(len(prop_var))]

# # Plot scree plot from PCA
# import matplotlib.pyplot as plt

# x_labels = ['PC{}'.format(i+1) for i in range(len(prop_var))]

# plt.plot(x_labels, prop_var, marker='o', markersize=6, color='skyblue', linewidth=2, label='Proportion of variance')
# plt.plot(x_labels, cum_var, marker='o', color='orange', linewidth=2, label="Cumulative variance")
# plt.legend()
# plt.title('Scree plot')
# plt.xlabel('Principal components')
# plt.ylabel('Proportion of variance')
# plt.show()

# =================================
# Plot the classes in 3 dims of pc?

# =================================
# LC
# read the test data
paths = glob.glob("./1/ds/test/*")
data_t = []
labels_t = []

for filePath in paths:
	x = re.findall("(\d+)", filePath);
	labels_t.append(int(x[-2]))

	x = []
	with open(filePath) as f:
		for i in range(32):
			line = f.readline();
			x.append(int(line, base=2))
	data_t.append(x)

data_t = np.array(data_t)
data_t = data_t / data_t.std()

# build models
from logistic_regression.classifier import lc

new_lables = [0 if l == 0 else 1 for l in labels]

print("making full model")
model_full = lc(32)
model_full.fit(data, new_lables, rounds=100)

print("making pca model")
model_pca = lc(31)
model_pca.fit(data[:, :-1], new_lables, rounds=100)
new_labels_t = [0 if l == 0 else 1 for l in labels_t]

# test
accuracy = 0
accuracy_pca = 0
for label, sample in zip(new_labels_t, data_t):
	prediction = model_full.predict(sample)
	prediction_pca = model_pca.predict(sample[:-1])
	
	if (prediction and label == 0) or (not prediction and label == 1):
		accuracy += 1

	if (prediction_pca and label == 0) or (not prediction_pca and label == 1):
		accuracy_pca += 1

print("full accuracy")
print (accuracy / len(new_labels_t))

print("pca accuracy")
print(accuracy_pca / len(new_labels_t))