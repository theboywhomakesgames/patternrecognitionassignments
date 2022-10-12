import numbers
import nltk
import glob
import string
import re

numbers_str = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
punc_string = "".join(string.punctuation)

def has_number(token):
	flag = True
	for n in numbers_str:
		if n in token:
			flag = False
			break
	return flag

def tokenize(ds_path):
	paths = glob.glob(ds_path)
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	all_the_words = []

	for address in paths:
		with open(address) as f:
			tokens = nltk.word_tokenize(f.read())
			tokens = list(filter(lambda token: token not in string.punctuation, tokens))

			tmp = []
			for token in tokens:
				splitted = re.split(r"[{}]+".format(punc_string), token)
				[tmp.append(new_token) for new_token in splitted]
			tokens = tmp

			tokens = list(filter(lambda token: token not in string.punctuation, tokens))
			tokens = list(filter(has_number, tokens))
			all_the_words = all_the_words + tokens

	dist = nltk.FreqDist(all_the_words)
	word_count = dict((w, f) for w, f in dist.items())
	word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))

	return word_count

def tokenize_doc(doc_path):
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	all_the_words = []

	with open(doc_path) as f:
		tokens = nltk.word_tokenize(f.read())
		tokens = list(filter(lambda token: token not in string.punctuation, tokens))

		tmp = []
		for token in tokens:
			splitted = re.split(r"[{}]+".format(punc_string), token)
			[tmp.append(new_token) for new_token in splitted]
		tokens = tmp

		tokens = list(filter(lambda token: token not in string.punctuation, tokens))
		tokens = list(filter(has_number, tokens))
		all_the_words = all_the_words + tokens

	dist = nltk.FreqDist(all_the_words)
	word_count = dict((w, f) for w, f in dist.items())
	word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))

	return word_count