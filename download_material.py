import os
import requests
# 数据来源
url='https://www.ximalaya.com/revision/play/v1/show?id=46104409&sort=0&size=30&ptype=1'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3724.8 Safari/537.36'}
res = requests.get(url,headers=headers)
res_json = res.json()
play_list = res_json['data']['tracksAudioPlay']
ALL_PATH = play_list[0]['albumName']
##print(play_list)
# print(ALL_PATH)
for i in play_list:
    #print(i['trackName'])
    #print(i['trackCoverPath'])
    #print(i['trackId'])
    # 获取文件信息 (标题 音乐路径 图片路径)
    url_title = i['trackName']
    url_id= i['trackId']
    url_cover_path = 'https:' + i['trackCoverPath']
    #请求json数据(src)
    m4a_url_json='https://www.ximalaya.com/revision/play/v1/audio?id={}&ptype=1'.format(url_id)
    data_json=requests.get(url=m4a_url_json,headers=headers).json()
    # print(data_json)
    m4a_url=data_json['data']['src']  #提取字典中src的内容
    #print(m4a_url)
    #保存数据
    data_m4a=requests.get(url=m4a_url,headers=headers).content
    name=url_title+'.wav'
    if not os.path.exists("dataset/郭德纲/"):
        os.makedirs("dataset/郭德纲/")
    with open('dataset/郭德纲/'+name,mode='wb') as f:
        f.write(data_m4a)
        print("保存成功:",name)
