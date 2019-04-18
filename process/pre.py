import codecs
import sys
import spacy
import pdb

def main(cn, en):
    nlp = spacy.load('en')
    save_cn = codecs.open(cn+".pos", 'w', encoding='utf-8')
    save_en_tok = codecs.open(en+".tok", 'w', encoding='utf-8')
    save_en_pos = codecs.open(en+".pos", 'w', encoding='utf-8')
    with codecs.open(cn, 'r', encoding='utf-8') as f:
        with codecs.open(en, 'r', encoding='utf-8') as g:
            cnt = 0
            for gline in g:
                cnt += 1
                if cnt % 1000 == 0:
                    print('processing line at : ', cnt)
                gline = gline.strip()
                fline = f.readline().strip()
                if "|" in gline:
                    print('skiping line at : ', cnt)
                    continue
                words = nlp(gline)
                g_tok_save_line = ' '.join([word.text.lower().strip() for word in words])
                #words_tok = nlp(g_tok_save_line)
                g_pos_save_line = ' '.join([word.text.lower().strip() + "|" + word.pos_.lower().strip() for word in words if word.pos_ != "SPACE"])
                save_cn.write(fline + "\n")
                save_en_tok.write(g_tok_save_line + "\n")
                save_en_pos.write(g_pos_save_line + "\n")
    save_cn.close()
    save_en_pos.close()
    save_en_tok.close()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
