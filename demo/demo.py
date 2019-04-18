import bottle
from bottle import get, post, request, response

from youdao import Youdao
from baidu import Baidu
from google import Google
from cmcm import E2C, C2E

import sys

reload(sys)
sys.setdefaultencoding('UTF-8')

from argparse import Namespace
import json
import os
import onmt
import time
import codecs

import torch
import argparse

parser = argparse.ArgumentParser(description='cm translate')
parser.add_argument('-c2e_model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-c2e_codes', default="",
                   help="bpe codes to be used")

parser.add_argument('-e2c_model', required=True,
                    help='Path to model .pt file')

parser.add_argument('-e2c_codes', default="",
                   help="bpe codes to be used")


parser.add_argument('-beam_size', type=int, default=12,
                    help='Beam size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action='store_true')
parser.add_argument('-gpu', type=int, default=-1,
                    help='gpu to be used')
parser.add_argument('-seprator', default='@@',
                    help="bpe seprator")

parser.add_argument('-alpha', type=float, default=1.0,
                    help="")
parser.add_argument('-beta', type=float, default=0.0,
                    help="")



opt = parser.parse_args()
opt.cuda = opt.gpu > -1
if opt.cuda:
    torch.cuda.set_device(opt.gpu)

opt.batch_size = 50
opt.n_best = 1

c2e_opt = opt
c2e_opt.bpe_codes = opt.c2e_codes
c2e_opt.model = opt.c2e_model
c2e = C2E(c2e_opt)

e2c_opt =  opt
e2c_opt.bpe_codes = opt.e2c_codes
e2c_opt.model = opt.e2c_model
e2c = E2C(e2c_opt)


#### youdao info
youdao_app_id = "5dcd671707dab45f"
youdao_key = "Z4h3RJgLdmsKro9148kxm13zzHh9YjkI"
youdao_app = Youdao(youdao_app_id, youdao_key)

baidu_app_id = "20160825000027412"
baidu_key = "nqZwgqSR74topEKu8MGL"
baidu_app = Baidu(baidu_app_id, baidu_key)

google_app = Google()
#######

###

def cmcmTran(doc, direction):
    if direction == 'e2c':
        rstr = e2c.translate(doc.strip())
    else:
        rstr = c2e.translate(doc.strip())
    return rstr.strip() 

def youdaoTran(app, doc, src, tgt):
    res = app.translate(doc.encode('utf-8'), src, tgt)
    return res

def baiduTran(app, doc, src="en", tgt="ch"):
    res = app.translate(doc.encode('utf-8'), src, tgt)
    return res

def googleTran(app, doc, src="en", tgt="ch"):
    res = app.translate(doc.encode('utf-8'), src, tgt)
    return res

@post('/trans')
def do_trans():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
    print "enter trans ... " 
    updateHistory = request.forms.get('updatehistory')
    if updateHistory == "1":
        history = ""
        if os.path.exists('update.log'):
            history = json.loads(codecs.open("update.log", 'r', encoding="utf-8").read().strip())
        else:
            history = "no logs now"
        json_res = json.dumps({"updatehistory" : history})
        return json_res
    translations = {}
    src = request.forms.get('query').strip()
    srcLan = request.forms.get('from').strip()
    tgtLan = request.forms.get('to').strip()
    engine = request.forms.get('engine').strip()
    #import pdb
    #pdb.set_trace()
    
    translations = ""

    print 'input src : ', src
    print 'src language : ', srcLan
    print 'tgt language : ', tgtLan


    if engine == "cmcm":
        if srcLan == "en" and tgtLan == "ch":
            direction = "e2c"
        elif srcLan == "ch" and tgtLan == "en":
            direction = "c2e"
        else:
            print "not valid direction"
        translations = cmcmTran(src, direction)
    elif engine == "google":
        translations = googleTran(google_app, src, srcLan, tgtLan)
    elif engine == "baidu":
        translations = baiduTran(baidu_app, src, srcLan, tgtLan)
    elif engine == "youdao":
        translations = youdaoTran(youdao_app, src, srcLan, tgtLan)
    else:
        print "bad engine"
        translations = "bad engine"
    json_res = json.dumps({"query":src, "trans" : translations, "engine" : engine})
    #import ipdb
    print "json res : ", json_res
    #ipdb.set_trace()
    #json_res = json.dumps({"src":src, "trans":translations})
    return json_res

#bottle.run(host='0.0.0.0', port=9090)
bottle.run(host='0.0.0.0', server="paste", port=9090)
#bottle.run(host='0.0.0.0', server="cherrypy", port=9090)
