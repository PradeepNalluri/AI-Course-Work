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
		if row[index] <= value:
			left_part.append(row)
		else:
			right_part.append(row)
	return left_part, right_part
def get_entropy_of_split(groups):
	entropy=0
	size=0.0
	for group in groups:
		size+=len(group)
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
		if p==0:
			p=0.0001
		if n==0:
			n=0.0001
		entropy+=normalzd_group_size*(-1*((n/size)*np.log2(n/size) + (p/size)*np.log2(p/size)))
	return entropy
def get_best_split(dataset):
	entropy=0
	n=0
	p=0
	for vect in dataset:
		if(vect[-1]==0):
			n+=1
		else:
			p+=1
	size=float(n+p)
	if n==0:
		n=0.0001
	if p==0:
		p=0.0001
	entropy=-1*((n/size)*np.log2(n/size) + (p/size)*np.log2(p/size))
	best_info_gain=-1
	best_f_index=0
	best_groups=[[],[]]
	for f_index in range(0,len(dataset[0])-1):
		for row in dataset:
			groups=test_split(f_index,row[f_index],dataset)
			entropy_of_split=get_entropy_of_split(groups)
			information_gain=entropy-entropy_of_split
			if(information_gain >= best_info_gain):
				best_info_gain = information_gain
				best_f_index = f_index
				best_groups = groups
	return {'best_information_gain':best_info_gain,'best_f_index':best_f_index
			,'best_groups':best_groups}

def build_tree(train, max_depth, min_size):
	root = get_best_split(train)
	make_child(root, max_depth, min_size, 1)
	return root

def build_terminal(group):
	count=0
	count1=0
	for row in group:
		if(row[-1]==0):
			count+=1
		else:
			count1+=1
	if(count>=count1):
		return 0.0
	else:
		return 1.0

def make_child(node, max_depth, min_size, depth):
	left, right = node['best_groups']
	del(node['best_groups'])
	if not left or not right:
		node['left'] = node['right'] = build_terminal(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = build_terminal(left), build_terminal(right)
		return
	if len(left) <= min_size:
		node['left'] = build_terminal(left)
	else:
		node['left'] = get_best_split(left)
		make_child(node['left'], max_depth, min_size, depth+1)
	if len(right) <= min_size:
		node['right'] = build_terminal(right)
	else:
		node['right'] = get_best_split(right)
		make_child(node['right'], max_depth, min_size, depth+1)


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
min_size=10
# predicted=decision_tree(train_set,test_set,max_depth,min_size)
# print predicted
# actual = [row[-1] for row in test_set]
# correct=0
# for i in range(len(actual)):
# 	if actual[i] == predicted[i]:
# 		correct += 1
# print correct / float(len(actual)) * 100.0

#splitting
k=3
split_len=int(train_len/k)
train_set_copy=train_set
folded_set=[]
for i in range(k):
	folded_set.append([])
	while(len(folded_set[i])<split_len and len(train_set_copy)>0):
		# print(len(folded_set[i]),split_len)
		index = rand.randrange(len(train_set_copy))
		folded_set[i].append(train_set_copy.pop(index))
test_acc=[]
for i in folded_set:
	predicted=decision_tree(i,test_set,max_depth,min_size)
	actual = [row[-1] for row in test_set]
	correct=0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	test_acc.append(correct / float(len(actual)) * 100.0)
print(test_acc)
