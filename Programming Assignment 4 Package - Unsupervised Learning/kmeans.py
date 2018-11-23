import pandas as pd
import xlrd as xl
import random
#Reading the Excel file
DataF=pd.read_excel("Data Sets.xlsx",sheet_name='Data Set 5')
x_data=DataF['X'].values
y_data=DataF['Y'].values
dataset=list(zip(x_data, y_data))

#Step 2
sort_4=[]
sort_1=[]
copy_dataset=dataset
count=0
while(1):
    sort_4.append([])
    a=random.sample(copy_dataset,4)
    for i in a:
        sort_4[count].append(copy_dataset.pop(copy_dataset.index(i)))
    sort_1.append(copy_dataset.pop(copy_dataset.index(random.sample(copy_dataset,1)[0])))
    count+=1
    if(len(copy_dataset)==5):
        sort_4.append([])
        a=random.sample(copy_dataset,4)
        for i in a:
            sort_4[count].append(copy_dataset.pop(copy_dataset.index(i)))
        sort_1.append(copy_dataset.pop(copy_dataset.index(copy_dataset[0])))
        break
print (sort_4)
# print (sort_1)
