import codecs
import spacy
import sys

def main(en_pos, en_bpe):
    saver = codecs.open(en_bpe + ".postag", 'w', encoding='utf-8')
    with codecs.open(en_pos, 'r', encoding='utf-8') as enPos, \
            codecs.open(en_bpe, 'r', encoding='utf-8') as enBpe:
        cnt = 0
        for posLine in enPos:
            cnt += 1
            posL = posLine.strip().split()
            bpeLine = enBpe.readline()
            bpeL = bpeLine.strip().split()

            posLL = [(tok.split('|')[0], tok.split('|')[1])  for tok in posL if tok.split('|')[1].lower() != "space"]

            posLL_assert = ' '.join([word[0].strip() for word in posLL])
            bpeLL_assert = bpeLine.replace('@@ ', '').strip()
            assert posLL_assert == bpeLL_assert, 'not match at line %i' % cnt

            idx = 0
            Res = []
            last_tok_is_bpe = False
            for i, tok in enumerate(bpeL):
                if not tok.strip().endswith('@@'):
                    Res.append(tok + "|" + posLL[idx][1])
                    idx += 1
                else:
                    tmp = tok.strip().replace('@@', '')
                    assert tmp in posLL[idx][0], 'tmp not in idx %i ' % idx
                    Res.append(tok + "|" + posLL[idx][1])
            #print(' '.join(Res))
            saver.write(' '.join(Res) + "\n")
    saver.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
