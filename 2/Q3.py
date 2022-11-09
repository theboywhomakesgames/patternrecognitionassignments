"""
Implement Quadratic Discriminant Analysis for dataset 1.
Display and comment the accuracy of the classifier in the training and test sets

constants are calculated for each class once
"""
import numpy as np
import glob
import re

def gaussian(x, mu, sigma):
	# calcualte the multi-variant guassian
	det = np.linalg.det(sigma)
	down = 2 * np.pi * np.sqrt(det)
	xc = x - mu
	xt = np.transpose(xc)
	inv = np.linalg.inv(sigma)
	e_power = -0.5 * (np.transpose(xc) @ inv @ xc)
	return np.exp(e_power) / down

def prob(classidx, x, sigmas, mus, priors):
	g = gaussian(x, mus[classidx], sigmas[classidx])
	return g * priors[classidx]

def predict(sample, sigmas, mus, priors):
	# calculate prob of belonging
	probs = np.zeros((10,))
	for i in range(10):
		probs[i] = prob(i, sample, sigmas, mus, priors)

	# choose class
	guessed_label = np.argmax(probs)
	return guessed_label

# read the data
paths = glob.glob("./1/ds/training_validation/*")
data = []
labels = []

for filePath in paths:
	x = re.findall("(\d+)", filePath);
	labels.append(x[-2])

	x = []
	with open(filePath) as f:
		for i in range(32):
			line = f.readline();
			x.append(int(line, base=2))
	data.append(x)

data = np.array(data)
data = data / data.std()

# seperate the data by label
classes = [[], [], [], [], [], [], [], [], [], []]
cur_label = ''
for label, sample in zip(labels, data):
	classes[int(label)].append(sample)

# calculate all the sigmas and mus and priors for different classes
sigmas = []
mus = []
priors = []

for cls in classes:
	cls = np.array(cls)
	cov = np.cov(np.transpose(cls))
	mu = np.mean(cls)
	priors.append(len(cls)/len(data))
	sigmas.append(cov)
	mus.append(mu)

# training data accuracy
accuracy = 0
for label, sample in zip(labels, data):
	if int(label) == predict(sample, sigmas, mus, priors):
		accuracy += 1
print("training set accuracy: " + str(accuracy/len(data)))

# read the test data
paths = glob.glob("./1/ds/test/*")
data = []
labels = []

for filePath in paths:
	x = re.findall("(\d+)", filePath);
	labels.append(x[-2])

	x = []
	with open(filePath) as f:
		for i in range(32):
			line = f.readline();
			x.append(int(line, base=2))
	data.append(x)

data = np.array(data)
data = data / data.std()

# test using the calculated sigmas
accuracy = 0
for label, sample in zip(labels, data):
	guess = predict(sample, sigmas, mus, priors)

	# evaluate correctness
	if(int(label) == guess):
		accuracy += 1

# print accuracy
accuracy /= len(data)
print("test set accuract: " + str(accuracy))