# _*_ coding: utf-8 _*_

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re
import codecs

import onmt

import torch
import nltk
from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer
from nltk.tokenize import sent_tokenize

from apply_bpe import BPE
from bosonnlp import BosonNLP
import jieba


PAT = re.compile(ur'[。？！]')


class CMTRANS(object):
    def __init__(self):
        pass

class E2C(object):
    def __init__(self, opt):
        self.opt = opt
        self.sep = opt.seprator + " "
        if opt.cuda:
            torch.cuda.set_device(opt.gpu)
        self.bpe = BPE(codecs.open(opt.bpe_codes, 'r', encoding="UTF-8"), opt.seprator, None, None)

        self.tokenizer = MosesTokenizer()
        self.detokenizer = MosesDetokenizer()
        self.translator = onmt.Translator(opt)

    def tokenDoc(self, doc):
        sentenceList = sent_tokenize(doc.strip())
        print 'e2c sentenceList : ', sentenceList
        tokens = []
        for sent in sentenceList:
            sent = sent.lower()
            sent = self.detokenizer.unescape_xml(self.tokenizer.tokenize(sent, return_str=True))
            if self.opt.bpe_codes != "":
                sent = self.bpe.segment(sent).strip()
            token = sent.split()
            tokens += [token]
        return tokens

    def translate(self, doc):
        batch = self.tokenDoc(doc)
        pred, _, _, _, _ = self.translator.translate(batch, None)
        rstr = ""
        #ipdb.set_trace()
        for idx in range(len(pred)):
            rstr += ''.join(' '.join(pred[idx][0]).replace(self.sep, '').split()) + "\n\n"
        print 'e2c rstr : ', rstr.strip()
        return rstr.strip() 

class C2E(object):
    """
    """
    def __init__(self, opt):
        self.opt = opt
        self.sep = opt.seprator + " "
        if opt.cuda:
            torch.cuda.set_device(opt.gpu)
        self.bpe = BPE(codecs.open(self.opt.bpe_codes, 'r', encoding="UTF-8"), self.opt.seprator, None, None)

        self.translator = onmt.Translator(opt)

        self.nlp = BosonNLP("NGhNiav2.16134.DvyEDmGzYd2S")

    def seg(self, doc):
        res = ""
        try:
            print "using boson....."
            boson_res = self.nlp.tag(l)
            res = boson[0]['word']
        except:
            res = jieba.cut(doc, cut_all=False)
        return " ".join(res)

    def truecase(self, text):
        text = text.encode('utf-8')
        truecase_sents = []
        tagged_sent = nltk.pos_tag([word.lower() for word in nltk.word_tokenize(text)])
        normalize_sent = [w.captitalize() if t in ['NN', 'NNS'] else w for (w, t) in tagged_sent]
        normalize_sent[0] = normalize_sent[0].capitalize()
        pretty_string = re.sub(" (?=[\.,'!?:;])", "", ' '.join(normalized_sent))
        return pretty_string

    def tokenDoc(self, doc):
        doc = doc.strip()
        sentenceList = re.split(PAT, doc.decode('utf-8'))
        assert len(sentenceList) >= 1
        if sentenceList[-1].strip() == "":
            sentenceList = sentenceList[:-1]
        punctuaList = re.findall(PAT, doc.decode('utf-8'))
        punctuaList += (len(sentenceList) - len(punctuaList)) * [' ']
        sents = [ sent + punc for (sent, punc) in zip(sentenceList, punctuaList)]
        sents = [ sent.strip() for sent in sents]
        print 'c2e sentenceList : ', sentenceList
        tokens = []
        for sent in sents:
            sent = sent.lower()
            #sent = self.detokenizer.unescape_xml(self.tokenizer.tokenize(sent, return_str=True))
            sent = self.seg(sent)
            if self.opt.bpe_codes != "":
                sent = self.bpe.segment(sent).strip()
            token = sent.split()
            tokens += [token]
        print 'c2e tokens : ', tokens
        return tokens

    def translate(self, doc):
        batch = self.tokenDoc(doc)
        pred, _, _, _, _ = self.translator.translate(batch, None)
        rstr = ""
        for idx in range(len(pred)):
            pred_sent = ' '.join(pred[idx][0]).replace(' @-@ ', '-').replace(self.sep, '')
            #pred_sent = self.truecase(pred_sent)
            pred_sent = pred_sent.capitalize()
            rstr += pred_sent + "\n"
        print 'c2e rstr : ', rstr.strip()
        return rstr.strip() 
