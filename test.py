
import numpy as np
def entropy(num_x,num_y):
    total=num_x+num_y
    return -1*((num_x/total)*np.log2(num_x/total) + (num_y/total)*np.log2(num_y/total) )
def information_gain(entropy):
    return 1-entropy
def gini_index(num_x_g,num_x_g_p,num_y_l,num_y_l_p):
    num_x_g_n=num_x_g-num_x_g_p
    num_y_l_n=num_y_l-num_y_l_p
    great_var=1-2*((num_x_g_n+num_x_g_p)/num_x_g)
    less_var=1-2*((num_y_l_n+num_y_l_p)/num_y_l)
    return ((great_var*num_x_g)+(less_var*num_y_l))/(num_x_g+num__l)
dataset=[]
with open('dataset.txt') as f:
    for line in f:
        dataset.append( [float(x) for x in line.split(',')] )
count=0
count1=0
for i in dataset:
    if(i[-1]==1):
        count1+=1
    else:
        count+=1
print("No of 0",count)
print("1's",count1)
print("Entropy:",entropy(count,count1))
