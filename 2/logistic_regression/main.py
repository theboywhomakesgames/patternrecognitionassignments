import classifier
import glob
import numpy as np

# import the ds
print("reding labels")
ds_paths = glob.glob("./ds/training_validation/*")
Y = [label[31] for label in ds_paths] # these numbers should be changed if re-ran in a different directory
Y = np.array(Y)

ds_test_path = glob.glob("./ds/test/*")
YT = [label[16] for label in ds_test_path] # these numbers should be changed if re-ran in a different directory
YT = np.array(YT)

def read_samples(ds_paths, Y):
	ds = []
	tmp_Y = []
	for path, label in zip(ds_paths, Y):
		if(label == "0" or label == "6"):
			with open(path) as f:
				ds.append(f.readlines())
				if(label == "0"):
					tmp_Y.append(0)
				else:
					tmp_Y.append(1)

				f.close()

	X = []
	for data in ds:
		x = []
		for line in data:
			x.append(int(line[:-1], 2))
		X.append(x)

	return X, tmp_Y

print("reading ds")
X, Y = read_samples(ds_paths, Y)
X = np.array(X)
Y = np.array(Y)

XT, YT = read_samples(ds_test_path, YT)
XT = np.array(XT)
YT = np.array(YT)

print("data read, sample:")
print(XT)
print(YT)

print("fitting model")
model = classifier.lc(32)
model.fit(X, Y)
print("fitted, testing...")

hits = 0
for test_data, test_label in zip(XT, YT):
	prediction = model.predict(test_data)
	if(prediction == test_label):
		hits += 1

print("accuracy:")
print(hits/YT.shape[0])