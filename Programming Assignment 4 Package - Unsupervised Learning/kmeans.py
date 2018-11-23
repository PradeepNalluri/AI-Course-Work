import pandas as pd
import xlrd as xl
import random
import math
def initU(c_size):
    U=[]
    for i in range(c_size):
        U.append([])
        for j in range(800):
            U[-1].append(random.uniform(0.0,1.0))
    return U

def calvi(M,Z,m):
    v=[]
    sumv=0
    for i in range(len(Z)):
        for j in range(len(M)):
            sumv+=math.pow(M[j],m)*Z[i][j]
        denm=0
        for i in M:
            denm+=math.pow(i,m)
        v.append(sumv/denm)
    return v



#Reading the Excel file
DataF=pd.read_excel("Data Sets.xlsx",sheet_name='Data Set 5')
x_data=DataF['X'].values
y_data=DataF['Y'].values
dataset=list(zip(x_data, y_data))

#Step 2
sort_4=[]
sort_1=[]
copy_dataset=dataset

while(1):
    a=random.sample(copy_dataset,4)
    for i in a:
        sort_4.append(copy_dataset.pop(copy_dataset.index(i)))
    sort_1.append(copy_dataset.pop(copy_dataset.index(random.sample(copy_dataset,1)[0])))
    if(len(copy_dataset)==5):
        a=random.sample(copy_dataset,4)
        for i in a:
            sort_4.append(copy_dataset.pop(copy_dataset.index(i)))
        sort_1.append(copy_dataset.pop(copy_dataset.index(copy_dataset[0])))
        break
ds=[[],[]]
for elem in sort_4:
    ds[0].append(elem[0])
    ds[1].append(elem[1])
U=initU(3)
m=1
V=[]
for i in U:
    V.append(calvi(i,ds,m))
print V
