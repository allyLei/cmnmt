from googletrans import Translator
import hashlib
import uuid


class Google(object):
    """
        google translation
    """
    def __init__(self):
        self.translator = Translator(service_urls=['translate.google.com'])
        self.langList = {}
        self.langList['ch'] = 'zh-CN'
        self.langList['en'] = 'en'

    def translate(self, query, src, tgt):
        try:
            res = self.translator.translate(query, dest=self.langList[tgt])
            return res.text
        except:
            return "google translation failed"

if __name__ == "__main__":
    google = Google()
    query = u"I have a dream!"
    res = google.translate(query, "en", "ch")
    print res

