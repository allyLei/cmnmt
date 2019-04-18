#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: songhongwei 
@ceate_time: 2017/10/18 19:41
@brief: postprocessing.py 对AI Challenger的语料进行后处理,将…替换为……
"""
import sys
import os
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)
import argparse
import codecs
import re


def parse_args(parser):
    #group = parser.add_mutually_exclusive_group()
    #group.add_argument("-p", "--predict", action="store_true")
    parser.add_argument("inp", help="input path")
    parser.add_argument("outp", help="output path")
    parser.add_argument("--src_input", default="../data/test_a.en", help="src_input path")
    args = parser.parse_args()
    return args

def processing_words(src_line, tgt_line):
    pos = 0
    pattern = re.compile('[a-zA-Z0-9_\-]+')
    while True:
        word = pattern.search(tgt_line, pos=pos)
        if not word:
            break
        match = re.search('(?<![\w\-])' + word.group() + '(?![\w\-])', src_line, re.I)
        replace_word = word.group()

        if match:
            replace_word = match.group()
            reason = "match src"
        elif len(word.group()) <= 3:
            replace_word = word.group().upper()
            reason = "upper"
        else:
            replace_word = word.group().capitalize()
            reason = "capitalize"
        tgt_line = tgt_line[:word.start()] + replace_word + tgt_line[word.end():]
        print("replace word,%s:%s\t%s" % (reason, word.group(),replace_word))
        pos = word.end()

    return tgt_line

def post_processing(args):
    inp = args.inp
    outp = args.outp
    src_input = args.src_input
    with codecs.open(inp, 'r', 'UTF-8', errors='ignore') as fi, \
        codecs.open(outp, 'w', 'UTF-8', errors='ignore') as fo, \
        codecs.open(src_input, 'r', 'UTF-8', errors='ignore') as f_src:
        while True:
            tgt_line = fi.readline()
            src_line = f_src.readline()
            if not src_line and not tgt_line:
                break
            # dot
            tgt_line = re.sub('…+', '……', tgt_line)
            tgt_line = re.sub('@-@', ' ', tgt_line)
            tgt_line = processing_words(src_line, tgt_line)
            fo.write(tgt_line)

def main(inp):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    args = parse_args(parser)
    post_processing(args)
    #src_line = u'34.00 mile(s) from Barrett Jackson Auction, the finest collector car auction in the world at Westworld in nearby Scottsdale'
    #tgt_line = u'距 barrett jackson 拍卖会 34.00 英里 ， 这 是 位于 斯科茨代尔 的 西方 世界 最 好 的 收藏 汽车 拍卖会 。abc.abc，son'
    #tgt_line = processing_words(src_line,tgt_line)
    #print(src_line)
    #print(tgt_line)

