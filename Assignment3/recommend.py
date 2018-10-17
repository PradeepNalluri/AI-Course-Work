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
