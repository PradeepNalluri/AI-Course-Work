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
def get_entropy_of_split(groups,classes):
	entropy=0
	size=0.0
	for group in groups:
		size+=len(group)
	for group in groups:
		normalzd_group_size=len(group)/size
		group_sum=0
		for clas in classes:
			count=0
			for vect in group:
				if vect[-1]==clas:
					count+=1
			if count==0:
				count=0.0001
			group_sum-=(count/size)*np.log2(count/size)
		entropy+=normalzd_group_size*group_sum
	return entropy
def get_best_split(dataset,classes):
	entropy=0
	group_sum=0
	size=float(len(dataset))
	for clas in classes:
		count=0
		for vect in dataset:
			if vect[-1]==clas:
				count+=1
		if count==0:
			count=0.0001
		group_sum-=(count/size)*np.log2(count/size)
	entropy+=group_sum
	best_info_gain=-1
	best_f_index=0
	best_groups=[[],[]]
	for f_index in range(0,len(dataset[0])-1):
		for row in dataset:
			groups=test_split(f_index,row[f_index],dataset)
			entropy_of_split=get_entropy_of_split(groups,[0,1])
			information_gain=entropy-entropy_of_split
			if(information_gain >= best_info_gain):
				best_info_gain = information_gain
				best_f_index = f_index
				best_groups = groups
				best_value=row[f_index]
	return {'best_value':best_value,'best_f_index':best_f_index
			,'best_groups':best_groups}

def build_tree(train, max_depth, min_size,classes):
	root = get_best_split(train,classes)
	make_child(root, max_depth, min_size, 1,classes)
	return root

def build_terminal(group,classes):
	max_count=0
	r_val=classes[0]
	for clas in classes:
		count=0
		for row in group:
			if row[-1]==clas:
				count+=1
		if(count>max_count):
			max_count=count
			r_val=clas
	return r_val
def make_child(node, max_depth, min_size, depth,classes):
	left, right = node['best_groups']
	del(node['best_groups'])
	if not left or not right:
		node['left'] = node['right'] = build_terminal(left + right,classes)
		return
	if depth >= max_depth:
		node['left'], node['right'] = build_terminal(left,classes), build_terminal(right,classes)
		return
	if len(left) <= min_size:
		node['left'] = build_terminal(left,classes)
	else:
		node['left'] = get_best_split(left,classes)
		make_child(node['left'], max_depth, min_size, depth+1,classes)
	if len(right) <= min_size:
		node['right'] = build_terminal(right,classes)
	else:
		node['right'] = get_best_split(right,classes)
		make_child(node['right'], max_depth, min_size, depth+1,classes)


def predict(node, row):
	if row[node['best_f_index']] < node['best_value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def decision_tree(train, test, max_depth, min_size,classes):
	tree = build_tree(train, max_depth, min_size,classes)
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
classes=[]
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
	if(vect[-1] not in classes):
		classes.append(vect[-1])
max_depth=5
min_size=10
#splitting
n=input("Press 1 for bagging else press anyting for without bagging:")
if(n==1):
	k=input("Number of folds:")
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
		predicted=decision_tree(i,test_set,max_depth,min_size,classes)
		actual = [row[-1] for row in test_set]
		correct=0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		test_acc.append(correct / float(len(actual)) * 100.0)
	print(max(test_acc))

else:
	##with out bagging
	predicted=decision_tree(train_set,test_set,max_depth,min_size,classes)
	actual = [row[-1] for row in test_set]
	correct=0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	print correct / float(len(actual)) * 100.0
