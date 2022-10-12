import glob
import tokenizer
import math

comp_path = "./ds1/train/comp.sys.ibm.pc.hardware/*"
compT_path = "./ds1/test/comp.sys.ibm.pc.hardware/*"
sci_path = "./ds1/train/sci.electronics/*"
sciT_path = "./ds1/test/sci.electronics/*"

print("tokenizing...")
comp_tokens = tokenizer.tokenize(comp_path)
sci_tokens = tokenizer.tokenize(sci_path)
print("tokens ready!")

print("counting...")
total_count_0 = 0
total_count_1 = 0

for key in comp_tokens:
	total_count_0 += comp_tokens[key]

for key in sci_tokens:
	total_count_1 += sci_tokens[key]
print("counts ready")

print("testing...")
comp_paths = glob.glob(compT_path)
sci_paths = glob.glob(sciT_path)

print("tokenizing set...")
compT_tokens = tokenizer.tokenize(compT_path)
sciT_tokens = tokenizer.tokenize(sciT_path)
total_count_0t = 0
total_count_1t = 0

print("counting test words...")
for key in compT_tokens:
	total_count_0t += compT_tokens[key]

for key in sciT_tokens:
	total_count_1t += sciT_tokens[key]

corrects = 0
for p in comp_paths:
	test_comp_tokens = tokenizer.tokenize_doc(p)

	prob0 = 0
	prob1 = 0
	for key in test_comp_tokens:
		if key in comp_tokens:
			prob0 += math.log(comp_tokens[key] / total_count_0)
		
		if key in sci_tokens:
			prob1 += math.log(sci_tokens[key] / total_count_1)

	if prob0 > prob1:
		corrects += 1

for p in sci_paths:
	test_sci_tokens = tokenizer.tokenize_doc(p)

	prob0 = 0
	prob1 = 0
	for key in test_sci_tokens:
		if key in comp_tokens:
			prob0 += math.log(comp_tokens[key] / total_count_0)
		
		if key in sci_tokens:
			prob1 += math.log(sci_tokens[key] / total_count_1)

	if prob0 < prob1:
		corrects += 1

print("done")
print(corrects/(len(comp_paths) + len(sci_paths)))