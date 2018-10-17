from operator import itemgetter #for sorting
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
    id=int(uitem[i][0])
    items[id]=str_list=map(int,feature_vect.split('|'))
dataset=list()
check=0
for row in udata:
    if(row[0]==check):
        dataset[check-1].append([row[0],items[row[1]],row[-2]]  )
    else:
        dataset.append([row[0],items[row[1]],row[-2]])
        check=row[0]

# for user_set in dataset:
#     test_len=int(0.3*len(user_set))
#     train_len=len(user_set)-test_len
#     count=0
#     test_set=[]
#     classes=[]
#     while(count<test_len):
#         rand_num=rand.randint(0,len(user_set)-1)
#         if(user_set[rand_num] in test_set):
#             pass
#         else:
#             test_set.append(user_set[rand_num])
#             count+=1
#     train_set=[]
#     count=0
#     for vect in user_set:
#         if vect in test_set:
#             count+=1
#             pass
#         else:
#             train_set.append(vect)
#     	if(vect[-1] not in classes):
#     		classes.append(vect[-1])
