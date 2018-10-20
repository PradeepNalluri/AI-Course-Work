from operator import itemgetter #for sorting
##Decision Tree Assignment
import math
from csv import reader
import random as rand
import numpy as np
import matplotlib.pyplot as plt
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

###################
def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

#Data FIle
fname="ml-100k/u.data"
with open(fname) as f:
    udata = f.readlines()
for i in range(len(udata)):
	udata[i]=map(int, udata[i].split())
udata = sorted(udata, key=itemgetter(0))

# Item File
fname="ml-100k/u.item"
with open(fname) as f:
    uitem = f.readlines()
items={}
for i in range(len(uitem)):
    feature_vect=str(uitem[i][-38:-1])
    uitem[i]=map(str,uitem[i].split('|'))
    id=float(uitem[i][0])
    items[id]=map(float,feature_vect.split('|'))
dataset=list()
check=0
for row in udata:
    if(row[0]==check):
        dataset[check-1].append([row[0],items[row[1]],row[-2]])
    else:
        dataset.append([[row[0],items[row[1]],row[-2]]])
        check=row[0]
avg_mae=0
avg_rmse=0
avg_precision=[]
avg_recall=[]
for user_set in dataset:
    avg_precision.append([])
    avg_recall.append([])
    copy_userset=user_set
    user_set=[]
    for i in copy_userset:
        uid=i.pop(0)
        if i not in user_set:
            user_set.append(i)
    test_len=int(0.3*len(user_set))
    train_len=len(user_set)-test_len
    count=0
    test_set=[]
    classes=[]
    while(count<test_len):
        rand_num=rand.randint(0,len(user_set)-1)
        if(user_set[rand_num] in test_set):
            pass
        else:
            test_set.append(user_set[rand_num])
            count+=1
    train_set=[]
    count=0
    for vect in user_set:
        if vect in test_set:
            count+=1
            pass
        else:
            train_set.append(vect)
    	if(vect[-1] not in classes):
    		classes.append(vect[-1])
	new_train_set=[]
	for i in range(len(train_set)):
		new_train_set.append([])
		for j in train_set[i][0]:
			new_train_set[i].append(j)
		new_train_set[i].append(float(train_set[i][-1]))
	new_test_set=[]
	for i in range(len(test_set)):
		new_test_set.append([])
		for j in test_set[i][0]:
			new_test_set[i].append(j)
		new_test_set[i].append(float(test_set[i][-1]))
    max_depth=9
    min_size=5
    predicted=decision_tree(train_set,test_set,max_depth,min_size,classes)
    # print predicted
    actual = [row[-1] for row in test_set]
    # print actual
    correct=0
    for i in range(len(actual)):
		correct += abs(actual[i]-predicted[i])
    mae=float(correct)/len(actual)
    avg_mae+=mae
    # print (uid,'mae',mae)
    correct=0
    for i in range(len(actual)):
		correct += abs(actual[i]-predicted[i])*abs(actual[i]-predicted[i])
    rmse=float(correct)/len(actual)
    avg_rmse+=math.sqrt(rmse)
    K=[5,10,15,20,25]
    actual_copy=list(actual)
    predicted_copy=list(predicted)
    for k in K:
        actual_copy.sort(reverse=True)
        predicted_copy.sort(reverse=True)
        # want_actaul=actual_copy[0:k]
        reccommended_set=predicted_copy[0:k]
        predicted_index_list=[]
        num_reccomend=0
        for i in range(len(predicted)):
			if predicted[i]>=3:
				num_reccomend+=1
        num_reccomend_relv=0
        for sort_act in reccommended_set:
			# dups=list_duplicates_of(actual,sort_act)
			pred_dups=list_duplicates_of(predicted,sort_act)
			for ped_dup in pred_dups:
				if actual[ped_dup]>3:
					# actual.pop(ped_dup)
					num_reccomend_relv+=1
					break
        if(num_reccomend_relv>num_reccomend):
             print(num_reccomend_relv,num_reccomend)
        if(num_reccomend!=0):
            avg_precision[uid-1].append(num_reccomend_relv*100.0/k)
            avg_recall[uid-1].append(num_reccomend_relv*100.0/len(reccommended_set))
			# print("Recall is %f for user %d @K %d",num_reccomend_relv*100.0/num_relavent,uid,k)
			# print("Precision is %f for user %d @K %d",num_reccomend_relv*100.0/num_reccomend,uid,k)
        else:
			pass
# print avg_mae/943,avg_rmse/943
aa_precision=[]
aa_recall=[]
for j in range(len(K)):
	suml=0
	for i in range(len(avg_precision)):
		try:
			suml+=avg_precision[i][j]
		except:
			suml+=0
	# print "Average Precision at k: ",K[j]," is ",sum/943
	aa_precision.append(suml/943)

for j in range(len(K)):
	suml=0
	for i in range(len(avg_recall)):
		try:
			suml+=avg_recall[i][j]
		except:
			suml+=0
	# print "Average Recall at k: ",K[j]," is ",sum/943
	aa_recall.append(suml/943)

plt.plot(K,aa_precision)
plt.show()
plt.plot(K,aa_recall)
plt.show()
