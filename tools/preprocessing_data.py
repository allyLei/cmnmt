#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: songhongwei 
@ceate_time: 2017/10/11 18:18
@brief: find_multi_word
"""
import sys
import os
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)
import argparse
import re

def parse_args(parser):
    parser.add_argument('-d', '--dict', default='data/multi_word.dict')
    parser.add_argument("--src_input", help="input path")
    parser.add_argument("--tgt_input", help="input path")
    parser.add_argument("--src_output", help="output path")
    parser.add_argument("--tgt_output", help="output path")
    parser.add_argument("--opensubtitle_src", help="opensubtitle_src input path")
    parser.add_argument("--opensubtitle_tgt", help="opensubtitle_tgt input path")
    parser.add_argument("--split_sentence", action='store_true', help="拆句")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--merge_sentence", action='store_true', help="合并句")
    group.add_argument("--merge_whole_sentence", action='store_true', help="合并完整句")
    parser.add_argument("--black_sentence", help="低质量句子")
    args = parser.parse_args()
    return args

multi_word_dict = None

def is_multi_word(word):
    result = False
    word = word.replace(' ', '')
    if word.lower() in multi_word_dict or re.findall('^\d+\-.*', word):
        #print("multi_word:", word)
        result = True
    else:
        pass
        #print("not multi_word:", word)
    return result


def is_subword(w):
    if w.isalnum():
        return True
    return False

horizontal_line = '-'
def get_multi_word(line, begin):
    start = len(line)  # multi_word include line[start]
    end = start  # exclude line[end]
    has_multi_word = False
    pos = line[begin:].find(horizontal_line)
    if pos == -1:
        return start, end, False
    pos += begin
    start = pos
    end = pos + 1
    prev_start = start
    prev_end = end
    lspace = rspace = 0
    word_size = 0
    i = j = pos
    i = i-1
    while i >= begin and line[i] == " ":
        i -= 1
        lspace += 1
    while i >= begin and (is_subword(line[i])):
        i -= 1
    if is_subword(line[i+1]):
        word_size += 1
    else:
        return start, end, False
    #import pdb;pdb.set_trace()
    while word_size <= 3 and j < len(line) and line[j] == "-":
        j = j + 1
        while j < len(line) and line[j] == " ":
            j += 1
            rspace += 1
        while j < len(line) and (is_subword(line[j])):
            j += 1
        if is_subword(line[j - 1]):
            word_size += 1
        else:
            return prev_start, prev_end, has_multi_word
        # 规则的连字词,不带多余空格的,直接返回。
        if lspace + rspace == 0 or is_multi_word(line[i + 1:j]) or word_size >= 4:
            prev_start = i + 1
            prev_end = j
            has_multi_word = True
        while j < len(line) and line[j] == " ":
            j += 1
            rspace += 1
    if has_multi_word:
        print("multi_word:", line[prev_start:prev_end])
    else:
        print("not multi_word:", line[i+1:j])
    return prev_start, prev_end, has_multi_word


def process_horizontal_line(src, tgt):
    if horizontal_line not in src+tgt:
        return src, tgt
    src_result = ""
    tgt_result = tgt
    has_multi_word = False
    i = 0
    while i < len(src):
        start, end, _has_multi_word = get_multi_word(src, i)
        if _has_multi_word:
            has_multi_word = _has_multi_word
        src_result += (src[i:start] + "".join(src[start:end].split()))
        i = end
    #import pdb;pdb.set_trace()
    # en include horizontal_line
    reason = ""
    if horizontal_line in src:
        # a. zh not.
        if horizontal_line not in tgt:
            # -是连字符
            if has_multi_word:
                reason = "-是连字符"
            # 是连句符
            else:
                src_result = src_result.replace('-', '^')
                reason = "-是连句符"
        else: # b. zh also include
            # 假设中文里的-也是连字符,去掉其两侧的多余空格
            if has_multi_word:
                #tgt_result = re.sub('(?<=-)\ ?', '', re.sub('\ ?(?=-)', '', tgt_result))
                #reason = "中文里的-也是连字符,去掉其两侧的多余空格"
                reason = "-是连字符"
                pass
            # 是分隔符
            else:
                reason = "中英句都出现了-但是不是连字符,英文替换为^"
                src_result = src_result.replace('-', '^')
                """
                i = src_result.find('-') - 1
                while i>=0 and src_result[i] == ' ':
                    i -= 1
                # 前一句没有结束标点的补充.
                pad = ' '
                zh_pad = ' '
                # 原生的半句话补充句点。
                if src_result.endswith('-') and src_result[i].isalnum() :
                    pad = '.'
                    zh_pad = u'。'
                    reason = "原生的半句话补充句点。"
                # 已有标点的不补充.
                elif not src_result[i].isalnum():
                    pad = ' '
                    zh_pad = ' '
                    reason = "已有标点的不补充."
                # 无法区分到底是连字符还是分隔符。默认替换-为空格
                else:
                    reason = "无法区分到底是连字符还是分隔符。默认替换-为空格"
                src_result = src_result.replace('-', pad)
                tgt_result = tgt_result.replace('-', zh_pad)
                """
    else: # c. zh must include
        # 包含数字的不替换-
        #if len(re.findall('\d', tgt_result)) == 0:
        #    tgt_result = tgt_result.replace('-', ' ')
        #    reason = "包含数字的不替换-"
        pass
    print(reason,":",src_result,"\t",tgt_result)
    return src_result, tgt_result

def split_sentence(src, tgt):
    src_match = re.match('([\w\s\']+,)([\w\s\']+[\.!\?])', src)
    # import pdb;pdb.set_trace()
    if src_match and src_match.group() == src and len(src_match.groups()) == 2 \
            and len(src_match.group(1).split()) >= 3 \
            and len(src_match.group(2).split()) >= 3:
        tgt_match = re.match(u'([^，。？！!\?,\.]+，)([^，。？！!\?,\.]+[。？！!\?])', tgt)
        if tgt_match and tgt_match.group() == tgt and len(tgt_match.groups()) == 2:
            print("split sentence:","%s\t%s" % (src_match.groups(), tgt_match.groups()))
            return [(src_match.group(1),tgt_match.group(1)),(src_match.group(2),tgt_match.group(2))]
    return []

def is_whole_single_sentence(src,tgt):
    src_match = re.match('[\w\s\']+[\.!\?]', src)
    if src_match and src_match.group() == src and len(src_match.group().split()) <= 20:
        tgt_match = re.match(u'[^，。？！!\?,\.]+[。？！!\?]', tgt)
        if tgt_match:
            return True
    return False

def process_dots(sentence):
    s = re.sub('\.{3,}', '…', sentence)
    s = re.sub('…{2,}','…', s)
    if s != sentence:
        print("normalize-dots:",sentence,s)
    return s


def filter_opensubtitle(src_line, tgt_line):
    def too_match_word(line):
        word = re.findall('[a-zA-Z]', line)
        if len(word) * 1.0 / len(line) > 0.5:
            return True
        return False

    def length_diff(src, tgt):
        src_len = len(src.split())
        if src_len > 4 and src_len * 1.0 / len(tgt) > 1.6:
            return True
        if len(re.findall('[a-zA-Z]', src)) / len(tgt) < 0.5:
            return True
        return False
    reason = ""
    # [**],(**)
    if not src_line or not tgt_line:
        reason = "only -"
    elif re.findall('\[.*\]', src_line) or re.findall('\[.*\]', tgt_line) or \
            re.findall('\(.*\)', src_line) or re.findall('\(.*\)', tgt_line) or \
            re.findall('（.*', tgt_line):
        reason = "[]"
    # #**
    elif src_line.startswith('#') or tgt_line.startswith('#') or tgt_line.startswith("*"):
        reason = '#'
    elif src_line in tgt_line:
        reason = 'src in tgt'
    elif too_match_word(tgt_line):
        reason = 'too match english'
    elif length_diff(src_line, tgt_line):
        reason = 'length not match'
    elif not re.match('^[a-zA-Z]', src_line):
        reason = 'not startswith english'
    if reason:
        print("filt %s:%s" % (reason, src_line+'\t'+tgt_line))
        return True
    return False

def main(args):
    global multi_word_dict
    multi_word_dict = set([word.strip().split('\t')[0].lower() for word in open(args.dict)])
    all_data = set()
    whole_single_sentences = []
    normal_single_sentences = []
    black_sentences = set()
    if args.black_sentence:
        black_sentences = set([line.strip() for line in open(args.black_sentence)])
    tgt_fi = None
    if args.tgt_input:
        tgt_fi = open(args.tgt_input)
    tgt_fo = None
    if args.tgt_output:
        tgt_fo = open(args.tgt_output, "w")

    with open(args.src_input) as src_fi, \
         open(args.src_output, "w") as src_fo:

        def processing(src_fi, src_fo, tgt_fi=None, tgt_fo=None, index=0, opensubtitle=False):
            while True:
                index += 1
                src_ori = src_fi.readline()
                if tgt_fi:
                    tgt_ori = tgt_fi.readline()
                else:
                    tgt_ori = ""
                if not src_ori and not tgt_ori:
                    break
                # 去重
                src = src_ori.replace('\t', ' ').strip()
                tgt = tgt_ori.replace('\t', ' ').strip()
                line = src + '\t' + tgt
                if line not in all_data:
                    all_data.add(line)
                else:
                    if args.tgt_input:
                        print("filt dump:", line)
                        continue
                    else:
                        pass

                # 清理^符号
                if '^' in src:
                    src_t = re.sub('\^ s(\W)', '\'s\g<1>', src)
                    if src_t == src:
                        if args.tgt_input:
                            print("filt ^:", line)
                            continue
                        else:
                            src = re.sub('\^', ' ', src)
                            print("replace ^ s to <blank> : ", src_t)
                    else:
                        print("replace ^ s to 's : ", src_t)
                        src = src_t

                # 过滤
                src_result = re.sub('^\-\s*', '', src)
                tgt_result = re.sub('^\-\s*', '', tgt)
                tgt_result = re.sub('\{fn华文仿宋.*\}','', tgt_result)
                if opensubtitle and filter_opensubtitle(src_result, tgt_result):
                    continue

                src_result, tgt_result = process_horizontal_line(src_result, tgt_result)
                src_result = process_dots(src_result)
                tgt_result = process_dots(tgt_result)

                src_fo.write("%s\n" % src_result.strip())

                if args.split_sentence:
                    half_sentence = split_sentence(src_result, tgt_result)
                    for half in half_sentence:
                        src_fo.write("%s\n" % half[0].strip())

                if not opensubtitle and args.merge_sentence or args.merge_whole_sentence:
                    if src_ori.strip() + '\t' + tgt_ori.strip() in black_sentences:
                        pass
                    else:
                        if args.merge_sentence:
                            whole_single_sentences.append((src_result, tgt_result))
                        elif args.merge_whole_sentence:
                            if is_whole_single_sentence(src_result, tgt_result):
                                whole_single_sentences.append((src_result, tgt_result))
                                # elif 4 <= len(src_result.split()) <= 30 and '-' not in src_result and '^' not in src_result:
                                #    normal_single_sentences.append((src_result, tgt_result))

                if tgt_fo:
                    tgt_fo.write("%s\n" % tgt_result.strip())
                    if args.split_sentence:
                        for half in half_sentence:
                            tgt_fo.write("%s\n" % half[1].strip())

                if index % 100000 == 0:
                    print("processing %s ..." % index)
            return index

        index = processing(src_fi=src_fi, src_fo=src_fo, tgt_fi=tgt_fi, tgt_fo=tgt_fo, index=0,opensubtitle=False)
        if args.opensubtitle_src:
            opensubtitle_src_fi = open(args.opensubtitle_src)
            opensubtitle_tgt_fi = open(args.opensubtitle_tgt)
            processing(src_fi=opensubtitle_src_fi, src_fo=src_fo, tgt_fi=opensubtitle_tgt_fi, tgt_fo=tgt_fo,
                       index=index, opensubtitle=True)



        if args.tgt_output and (args.merge_sentence or args.merge_whole_sentence):
            import random
            random.seed(2017)
            random.shuffle(whole_single_sentences)
            #random.shuffle(normal_single_sentences)
            half_length = int(len(whole_single_sentences)/2)
            for idx in range(half_length):
                merge_src = whole_single_sentences[idx][0].strip() + ' ' +\
                            whole_single_sentences[half_length+idx][0].strip()
                merge_tgt = whole_single_sentences[idx][1].strip() + ' ' +\
                            whole_single_sentences[half_length+idx][1].strip()
                src_fo.write("%s\n" % merge_src)
                tgt_fo.write("%s\n" % merge_tgt)
                print("merge sentence: %s\t%s" % (merge_src, merge_tgt))

            #normal_sentences_len = len(normal_single_sentences)
            #p = 0
            #copy_num = 3
            #for first_half in whole_single_sentences:
            #    first_half_src, first_half_tgt = first_half
            #    for _ in range(copy_num):
            #        latter_src, latter_tgt = normal_single_sentences[p % normal_sentences_len]
            #        p += 1
            #        merge_src = first_half_src + ' ' + latter_src
            #        merge_tgt = first_half_tgt + ' ' + latter_tgt
            #        src_fo.write("%s\n" % merge_src)
            #        tgt_fo.write("%s\n" % merge_tgt)
            #        print("merge sentence: %s\t%s" % (merge_src, merge_tgt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    args = parse_args(parser)
    main(args)

