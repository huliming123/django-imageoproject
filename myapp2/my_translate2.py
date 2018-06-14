import requests
import json
def zhong_ying():
    while 1:
        my_sentence = input('输入:')
        url = 'http://fanyi.baidu.com/basetrans'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Mobile Safari/537.36'}
        form_data = {'query': my_sentence, 'from': 'zh', 'to': 'en'}
        try:
            html = requests.post(url, data=form_data, headers=headers).text
            html = json.loads(html)
            print(html.get('trans')[0].get('dst'))
        except Exception :
            print('结束啦!!!!')
        if my_sentence=='':
            return
def fanyi(sentencss):
        url = 'http://fanyi.baidu.com/basetrans'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Mobile Safari/537.36',
        }
        form_data = {'from': 'en', 'to': 'zh', 'query': sentencss}
        try:
            my_dic = requests.post(url, data=form_data, headers=headers).text
            my_dic = json.loads(my_dic)
            my_sentences=my_dic['trans'][0].get('dst')
            return my_sentences
        except Exception as e:
            print('结束')

if __name__=='__main__':
    your_chances=input('选择1进行中文翻译，选择2进行英文翻译,请开始您的选择:')
    if your_chances=='1':
        zhong_ying()
    if your_chances=='2':
        yingwen_zhongwen()
