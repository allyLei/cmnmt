#-*- encoding:utf-8 -*-
import codecs
import uuid
import urllib
import json
import hashlib
import requests




class Youdao(object):
    """
       youdao translate
    """
    def __init__(self, appid, key):
        self.appid = appid
        self.key = key
        self.langList = {}
        self.initLang()
    def initLang(self):
        self.langList['ch'] = 'zh-CHS'
        self.langList['en'] = 'EN'
        self.langList['ko'] = 'ko'

    def generateParams(self, query, src, tgt):
        params = {}
        #query = urllib.quote(str(query))
        salt = uuid.uuid4().hex
        sign = hashlib.md5(self.appid+query+salt+self.key).hexdigest()
        params['q'] = query
        params['sign'] = sign
        params['salt'] = salt
        params['appKey'] = self.appid
        params['from'] = self.langList[src]
        params['to'] = self.langList[tgt]
        return params

    def translate(self, query, src, tgt):
        print "entering youdao translate"
        assert src in self.langList.keys()
        assert tgt in self.langList.keys()
        url = "http://openapi.youdao.com/api"
        params = self.generateParams(query, src, tgt)
        try:
            respond = requests.post(url, params=params).json()
            #import ipdb
            #ipdb.set_trace()
            if respond['errorCode'] == u'0':
                return respond['translation'][0]
            else:
               return "translation failed"
        except:
            return "exception and translation failed"

if __name__ == "__main__":
    query = u"tang dynasty"
    youdao = Youdao("5dcd671707dab45f", "Z4h3RJgLdmsKro9148kxm13zzHh9YjkI")
    res = youdao.translate(query.encode('utf-8'), 'en', 'ch')
    print  "res : ", res
