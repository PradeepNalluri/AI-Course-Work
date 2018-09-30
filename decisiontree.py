from csv import reader
import random as rand
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

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = best_split(train)
	split(root, max_depth, min_size, 1)
	return root
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini
# Split a dataset based on an attribute and an attribute value
def temp_split(index, value, dataset):
	left_part, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left_part.append(row)
		else:
			right.append(row)
	return left_part, right
# Select the best split point for a dataset
def best_split(train_set):
	outcomes = list(set(row[-1] for row in train_set))
	b_index, b_value, b_score, b_groups = rand.randint(1,len(train_set)), rand.randint(1,len(train_set)), rand.randint(1,len(train_set)), None
	for index in range(len(train_set[0])-1):
		for row in train_set:
			groups = temp_split(index, row[index], train_set)
			gini = gini_index(groups, outcomes)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
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
		node['left'] = best_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = best_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

#Decision tree
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

#file name
filename = 'dataset.txt'
dataset = get_data(filename)
for i in range(len(dataset[0])):
	conv_float(dataset, i)
test_len=int(0.1*len(dataset))
train_len=len(dataset)-test_len
count=0
test_set=[]
while(count<test_len):
    rand_num=rand.randint(0,len(dataset))
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
max_depth = 5
min_size = 9
predicted= decision_tree(train_set, test_set,max_depth,min_size)
correct = 0
actual = [row[-1] for row in test_set]
for i in range(len(actual)):
	if actual[i] == predicted[i]:
		correct += 1
print correct / float(len(actual)) * 100.0
#/len(test_set))*100
