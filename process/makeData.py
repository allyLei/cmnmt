import argparse
import numpy as np
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('-src', type=str, required=True, help="")
parser.add_argument('-tgt', type=str, required=True, help="")
parser.add_argument('-src_train', type=str, required=True, help="")
parser.add_argument('-tgt_train', type=str, required=True, help="")
parser.add_argument('-src_valid', type=str, required=True, help="")
parser.add_argument('-tgt_valid', type=str, required=True, help="")
parser.add_argument('-size', type=int, default=2000, help="")
parser.add_argument('-seed', type=int, default=1937, help="")
opt = parser.parse_args()

def main():
    src_train = codecs.open(opt.src_train, 'w', encoding='UTF-8')
    tgt_train = codecs.open(opt.tgt_train, 'w', encoding='UTF-8')
    src_valid = codecs.open(opt.src_valid, 'w', encoding='UTF-8')
    tgt_valid = codecs.open(opt.tgt_valid, 'w', encoding='UTF-8')
    with codecs.open(opt.src, 'r', encoding='UTF-8') as f, \
            codecs.open(opt.tgt, 'r', encoding='UTF-8') as g:
        cnt = 0
        for fline in f:
            cnt += 1
        f.seek(0)
        print('there are ', cnt, ' lines')
        np.random.seed(opt.seed)
        choice = np.random.randint(0, cnt, opt.size)
        idx = 0
        for fline in f:
            fline = fline
            gline = g.readline()
            if idx in choice:
                src_valid.write(fline)
                tgt_valid.write(gline)
            else:
                src_train.write(fline)
                tgt_train.write(gline)
            idx += 1
            if idx % 1000 == 0:
                print('Processing line ', idx)
    src_train.close()
    tgt_train.close()
    src_valid.close()
    tgt_valid.close()
if __name__ == "__main__":
    main()
