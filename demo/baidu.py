#-*- encoding:utf-8 -*-
import requests
import hashlib
import uuid


class Baidu(object):
    """
        baidu translation
    """
    def __init__(self, appid, key):
        self.appid = appid
        self.key = key
        self.langList = {}
        self.initLang()

    def initLang(self):
        self.langList['ch'] = 'zh'
        self.langList['en'] = 'en'
        self.langList['auto'] = 'auto'

    def translate(self, query, src=None, tgt=None):
        assert src in self.langList.keys()
        assert tgt in self.langList.keys()

        url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'

        params = self.generateParams(query, src, tgt)
        try:
            respond = requests.post(url, params=params).json()
            return respond['trans_result'][0]['dst']
        except:
            return "baidu translation failed"

    def generateParams(self, query, src, tgt):
        params = {}
        salt = uuid.uuid4().hex
        sign = hashlib.md5(self.appid+query+salt+self.key).hexdigest()
        params['q'] = query
        params['from'] = self.langList[src]
        params['to'] = self.langList[tgt]
        params['appid'] = self.appid
        params['salt'] = salt
        params['sign'] = sign
        return params
if __name__ == "__main__":
    appid = "20160825000027412"
    key = "nqZwgqSR74topEKu8MGL"
    baidu = Baidu(appid, key)
    doc = u"我是一个中国人!"
    res = baidu.translate(doc.encode('utf-8'), 'ch', 'en')
    print res
