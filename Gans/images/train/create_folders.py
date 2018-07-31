import os
instances = []
for i in range(1,25):
    if(i<10):
        instances.append("0"+str(i))
    else:
        instances.append(str(i))
for each in instances:
    if not os.path.exists(each):
        os.makedirs(each)
