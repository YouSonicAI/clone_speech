import os
a=[]
# b=[]
def read_directory(dir_name):
    # for filename in os.listdir(dir_name):
    #     a.append(filename)
    a=os.listdir(dir_name)
    for i in a:
        if i=='.ipynb_checkpoints':#删除隐藏文件
            a.remove(i)
    print(a)
   
    f=open('label_list.txt','w')
    for line in a:
        f.write(line+'\n')
    f.close()

read_directory('dataset')

