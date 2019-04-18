#-*- encoding:utf-8 -*-
import codecs
import uuid
import urllib
import json
import hashlib
import requests
import time




class Tencent(object):
    """
       tencent translate
    """
    def __init__(self, appid, key):
        self.secretId = appid
        self.key = key
        self.langList = {}
        self.initLang()
    def initLang(self):
        self.langList['ch'] = 'zh'
        self.langList['en'] = 'en'
        self.langList['ko'] = 'ko'

    def generateParams(self, query, src, tgt):
        params = {}
        #query = urllib.quote(str(query))
        salt = uuid.uuid4().hex
        #sign = hashlib.md5(self.appid+query+salt+self.key).hexdigest()
        params['Action'] = "TextTranslate"
        params['Nonce'] = 3677
        params['SecretId'] = self.key
        params['Timestamp'] = int(time.time())
        params['source'] = self.langList[src]
        params['sourceText'] = query
        params['target'] = self.langList[tgt]

        signature = "GETcvm.api.qcloud.com/v2/index.php?"
        return params

    def translate(self, query, src, tgt):
        print "entering tencent translate"
        assert src in self.langList.keys()
        assert tgt in self.langList.keys()
        url = "https://tmt.api.qcloud.com/v2/index.php"
        params = self.generateParams(query, src, tgt)
        params_str = ""
        for key in sorted(params):
            params_str = params_str + "&" + key + "=" + str(params[key])
        params_str = params_str.strip("&")
        try:
            import pdb
            pdb.set_trace()
            #respond = requests.post(url, params=params).json()
            url += "?" + params_str
            respond = requests.get(url).json()
            #import ipdb
            #ipdb.set_trace()
            if respond['code'] == u'0':
                import pdb
                pdb.set_trace()
                return respond['targetText']
            else:
               return "translation failed"
        except:
            return "exception and translation failed"

if __name__ == "__main__":
    query = u"tang dynasty"
    tencent = Tencent("AKIDlQuBGAhZyCE4kM7267OabVGsIYfaDqFV", "DE3g6KLAmGcnmZBcv42oNpBLJ4d0FJBu")
    res = tencent.translate(query.encode('utf-8'), 'en', 'ch')
    print  "res : ", res
