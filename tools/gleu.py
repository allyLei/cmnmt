#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: songhongwei 
@ceate_time: 2017/10/31 18:29
@brief: gleu
"""
import sys
import os
import argparse
from nltk.util import ngrams
from collections import Counter

def parse_args(parser):
    group = parser.add_mutually_exclusive_group()
    #group.add_argument("-p", "--predict", action="store_true")
    parser.add_argument("inp_tgt", help="input path")
    parser.add_argument("inp_hyp", help="input path")
    #parser.add_argument("-v", "--value", type=float, default=0.5, help="threshold value")
    args = parser.parse_args()
    return args


def main(inp_tgt, inp_hyp):
    with open(inp_tgt) as ft, open(inp_hyp) as fh:
        while True:
            lt = ft.readline()
            lh = fh.readline()
            if not lt and not lh:
                break
            print(gleu(lt.lower().strip().split(), lh.lower().strip().split()))

def gleu(tgt, hyp):
    tgt_gram_counter, hyp_gram_counter = Counter(), Counter()
    for i in range(1, 5):
        tgt_gram_counter.update(ngrams(tgt, i))
        hyp_gram_counter.update(ngrams(hyp, i))
    common_gram_counter = Counter()
    for gram in hyp_gram_counter.keys():
        common_gram_counter[gram] = min(tgt_gram_counter.get(gram, 0), hyp_gram_counter[gram])
    common_gram_count = sum(common_gram_counter.values())*1.0
    #print("tgt_gram_counter:",tgt_gram_counter,"hyp_gram_counter:",hyp_gram_counter,"common_gram_counter:",common_gram_counter)
    try:
        return min(common_gram_count/sum(hyp_gram_counter.values()), common_gram_count/sum(tgt_gram_counter.values()))
    except:
        return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    args = parse_args(parser)
    main(args.inp_tgt, args.inp_hyp)
    exit(0)
    tgt = u"我 把 钱 还 你 ！ - 我 不 知 道 。".split()
    hyp = u"我 把 钱 还 你 ！ - 我 不 知 道  我 把 钱 还 你 ！ - 我 不 知 道 ".split()
    print(gleu(tgt, hyp))
    hyp = u"我 把 你 的 钱 还 给 你 ! 我 不 知 道 。".split()
    print(gleu(tgt, hyp))

