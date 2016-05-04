#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
import jieba

def remove_punc(text):
    signs = ["？", "！", "。", "，", "“", "”", "；","：", "、", "‘", "’", ".", "?",",",".", '…']
    for s in signs:
        text = text.replace(s, "")
    return text
def conv2list(text, o):
    seg_list = jieba.cut(text)
    o.write("BOS\tNN\tO\n")
    for word in seg_list:
        o.write(word.encode('utf8')+"\tNN\tE\n")
    o.write("EOS\tNN\tO\n\n")

text = raw_input()
o1 = open("./CSPPT/test/tmp/input", "w")
o2 = open("./CSPPT/test/tmp/input.list", "w")
o1.write(text)
o1.close()
text_nopunc = remove_punc(text)
conv2list(text_nopunc,o2)
o2.close()

