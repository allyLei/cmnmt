import re
import ipdb

NUM = "_NUM_"

def index(a):
    return chr(ord(a)+1)

def preprocessing(sentence):
    num_dict = {}
    num_pat = re.compile(r"0\.\d+|[1-9]\d*\.\d+|\d+")
    numbers = re.findall(num_pat, sentence)
    res = sentence
    idx = 'a'
    for num in numbers:
        res = res.replace(num, NUM+idx)
        idx = index(idx)

    idx = 'a'
    rep = 1
    for num in numbers:
        key = "_NUM_" + str(rep) + "_"
        res = res.replace("_NUM_"+idx, key)
        num_dict[key] = num
        rep = rep + 1
        idx = index(idx)
    return res, num_dict
def postprocessing(sentence, num_dict):
    s = sentence
    for key, val in num_dict.items():
        if key in s:
            s = s.replace(key, val)
    return s

if __name__ == "__main__":
    sentence = "I have 100.122 apple and 2 banans"
    #sentence = "I have 100,122 apple and 2 banans"
    replace, num_dict = preprocessing(sentence)
    bak = postprocessing(replace, num_dict)
    print "orgin : ", sentence
    print "replaced : ", replace
    print "replace dict : ", num_dict
    print "replace back : ", bak
