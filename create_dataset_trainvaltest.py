import os
import initialize_model
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



c=[]
d=[]
n=[]
label=[]
def file_name(file_dir):
    for filename in os.walk(file_dir):
        c.append(filename[0])
        d.append(filename[2])
    # print(c)
    # print(d)
    m=map(list,zip(c,d))
    for x in m:
        n.append(list(x))
    # print(n)
    n.pop(0)
    print(n)
    for i in n:
        for j in range(len(i[1])):
            label.append(i[0]+str("/")+i[1][j]+str("  ")+str(n.index(i)))
    # print(len(label))
    f=open('train_list.txt','w')
    for line in label:
        f.write(line+'\n')
    f.close()
file_name('dataset')


#划分数据集
from sklearn.model_selection import train_test_split

def train_test_val_split(data, ratio_train, ratio_test, ratio_val):
    train, middle = train_test_split(data, train_size=ratio_train, test_size=ratio_test + ratio_val)
    ratio = ratio_val/(1-ratio_train)
    test, validation = train_test_split(middle, test_size=ratio)
    return train, test, validation
train, test, validation = train_test_val_split(label, 0.6, 0.2, 0.2)
print('训练集为：',train)#18条数据
# print(len(train))
print('测试集为：',test)#6条数据
# print(len(test))
print('验证集为：',validation)#6条数据
data_train=[]
dt=[]
for i in train:
    words = i.split()
    secondwords = iter(words)
    next(secondwords)
    output = [' '.join((first, second))  for first, second in zip(words, secondwords)]
    data_train.append(output)
# print(data_train)
for i in range(len(data_train)):
    dt.append(data_train[i][0])
print(dt)
dt.pop(15)
for i in dt:
    print(i)

import paddle
from paddlehub.finetune.trainer import Trainer

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = Trainer(speaker_encoder, optimizer, checkpoint_dir='a')
trainer.train(, epochs=2, batch_size=16, eval_dataset=flowers_validate, log_interval=10, save_interval=1)


for i in dt:
    ref_audio_path =i#传参
    #ipd.Audio(filename=ref_audio_path)#下载到本地，对音频进行后期处理
    mel_sequences = p.extract_mel_partials(p.preprocess_wav(ref_audio_path))
    print("mel_sequences: ", mel_sequences.shape)
    with paddle.no_grad():
        embed = speaker_encoder.embed_utterance(paddle.to_tensor(mel_sequences))
    print("embed shape: ", embed.shape)