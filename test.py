from csv import reader
import random as rand
import numpy as np
#Dataset processing
def get_data(filename):
	file = open(filename, "rb")
	lines = reader(file)
	dataset = list(lines)
	return dataset

#Convert to floats
def conv_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def test_split(index, value, dataset):
	left_part, right_part = list(), list()
	for row in dataset:
		if row[index] < value:
			left_part.append(row)
		else:
			right_part.append(row)
	return left_part, right_part
def get_entropy_of_split(groups,size):
	entropy=0
	for group in groups:
		normalzd_group_size=len(group)/size
		n=0
		p=0
		group_sum=0
		for vect in group:
			if vect[-1]==1:
				p+=1
			else:
				n+=1
		total=len(group)
		size=float(size)
		if p==0:
			p=size
		if n==0:
			n=size
		entropy+=normalzd_group_size*(-1*((n/size)*np.log2(n/size) + (p/size)*np.log2(p/size)))
	return entropy
def get_best_split(dataset,size):
	entropy=0
	size=float(size)
	n=0
	p=0
	for vect in dataset:
		if(vect[-1]==0):
			n+=1
		else:
			p+=1
	entropy=-1*((n/size)*np.log2(n/size) + (p/size)*np.log2(p/size))
	best_info_gain=-1000
	best_f_index=0
	best_groups=None
	for f_index in range(0,len(dataset[0])-1):
		for row in dataset:
			groups=test_split(f_index,row[f_index],dataset)
			entropy_of_split=get_entropy_of_split(groups,len(dataset))
			information_gain=entropy-entropy_of_split
			if(information_gain > best_info_gain):
				best_info_gain = information_gain
				best_f_index = f_index
				best_groups = groups
	return {'best_information_gain':best_info_gain,'best_f_index':best_f_index
			,'best_groups':best_groups}

def build_tree(train, max_depth, min_size):
	root = get_best_split(train,len(train))
	split(root, max_depth, min_size, 1,len(train))
	return root

def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth,size):
	left, right = node['best_groups']
	del(node['best_groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_best_split(left,size)
		split(node['left'], max_depth, min_size, depth+1,size)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_best_split(right,size)
		split(node['right'], max_depth, min_size, depth+1,size)

def predict(node, row):
	if row[node['best_f_index']] < node['best_information_gain']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)


filename = 'dataset.txt'
dataset = get_data(filename)
for i in range(len(dataset[0])):
	conv_float(dataset, i)
test_len=int(0.1*len(dataset))
train_len=len(dataset)-test_len
count=0
test_set=[]
while(count<test_len):
    rand_num=rand.randint(0,len(dataset)-1)
    if(dataset[rand_num] in test_set):
        pass
    else:
        test_set.append(dataset[rand_num])
        count+=1
train_set=[]
count=0
for vect in dataset:
    if vect in test_set:
        count+=1
        pass
    else:
        train_set.append(vect)
max_depth=5
min_size=9
predicted=decision_tree(train_set,test_set,max_depth,min_size)
actual = [row[-1] for row in test_set]
correct=0
for i in range(len(actual)):
	if actual[i] == predicted[i]:
		correct += 1
print correct / float(len(actual)) * 100.0
