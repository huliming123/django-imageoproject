import requests
from bs4 import BeautifulSoup
def fanyi(sentences):
    url = 'http://m.youdao.com/translate'
    form_data = {'inputtext': sentences, 'type': 'AUTO'}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}
    try:
        html = requests.post(url, data=form_data, headers=headers).text
        soup = BeautifulSoup(html, 'lxml')
        my_sent=soup.select('#translateResult > li')[0].text
        return my_sent
    except Exception as e:
        pass
if __name__=='__main__':
    fanyi()


