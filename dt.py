import numpy as np
def entropy(dataset):
    count=0
    count1=0
    for i in dataset:
        if(i[-1]==1):
            count1+=1
        else:
            count+=1
    total=count+count1
    return -1*((count/total)*np.log2(count/total) + (count1/total)*np.log2(count1/total) )
def information_gain(dataset,child_set):
    entropy_current=entropy(dataset)
    print(entropy_current)
    entropy_child=0
    for i in child_set:
        entropy_child+=(len(i)/len(dataset))*entropy(i)
    return entropy_current-entropy_child
dataset=[]
with open('dataset.txt') as f:
    for line in f:
        dataset.append( [float(x) for x in line.split(',')] )
left_set=[]
right_set=[]
child_set=[]
for row in dataset:
    if row[0]>1:
        left_set.append(row)
    else:
        right_set.append(row)
child_set.append(left_set)
child_set.append(right_set)
IG=information_gain(dataset,child_set)
print(IG)
