import requests
import xml
useragent={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'}
url='https://vipreader.qidian.com/chapter/1033390924/715570443/'
resterpon=requests.get(url=url,headers=useragent).text
print(resterpon)