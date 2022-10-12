import enum
import glob
import numpy as np
import knn
import random

# import the ds
print("reding labels")
ds_paths = glob.glob("./ds/training_validation/*")
Y = [label[31] for label in ds_paths] # these numbers should be changed if re-ran in a different directory
Y = np.array(Y)

ds_test_path = glob.glob("./ds/test/*")
YT = [label[16] for label in ds_test_path] # these numbers should be changed if re-ran in a different directory
YT = np.array(YT)

def read_samples(ds_paths):
	ds = []
	for path in ds_paths:
		with open(path) as f:
			ds.append(f.readlines())
			f.close()

	X = []
	for data in ds:
		x = []
		for line in data:
			x.append(int(line[:-1], 2))
		X.append(x)

	return X

print("reading ds")
X = read_samples(ds_paths)
X = np.array(X)

XT = read_samples(ds_test_path)
XT = np.array(XT)

print("shuffling")
for i in range(len(Y)):
	first = random.randrange(0, len(Y))
	second = random.randrange(0, len(Y))
	X[first], X[second] = X[second], X[first]
	Y[first], Y[second] = Y[second], Y[first]

print("printing the output")
print(XT)

print("starting k-point")
scores = []

for i in range(1, 11):
	print("k = {}".format(i))

	model = knn.KNN(i)
	model.prepare(X, Y)

	# 5-fold
	score = 0
	fold_size = len(Y) // 5
	for fold_idx in range(5-1):
		hits = 0
		for idx in range(fold_idx*fold_size, fold_idx+1*fold_size):
			prediction = model.find_label(idx)
			if(prediction == Y[idx]):
				hits += 1

		cur_score = hits/(fold_idx+1*fold_size - fold_idx*fold_size)
		score += cur_score
	score /= 5
	print("accuracy for {} is {}".format(i, score))
	scores.append(score)
	
print("best k for knn is:")
print(np.argmax(scores))