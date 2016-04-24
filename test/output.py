#!/usr/bin/python
# -*- coding:utf8 -*-

import sys
def getline(lines):
    words = []
    labels = []
    for line in lines:
        line = line.split()
        words.append(line[0])
        labels.append(line[-1])
    return words, labels

i1 = open("test/tmp/input.list", "r")
i2 = open("test/tmp/input.list.result", "r")
i = open("test/tmp/input", "r")

text = i.read()
lines1 = i1.readlines()[1:-2]
lines2 = i2.readlines()[2:-2]

if (len(lines1) != len(lines2)):
    print len(lines1), len(lines2)
    print "Error, NOT EQUAL"
    exit()
words1, labels1 = getline(lines1)
words2, labels2 = getline(lines2)


output = ""
for i in range(len(words2)):
#    print words1[i], words2[i]
    output += words1[i]
    if labels2[i] == "EW":
        output += "？"
    elif labels2[i] == "EG":
        output += "！"
    elif labels2[i] == "ED":
        output += "，"
    elif labels2[i] == "EJ":
        output += "。"
if output[-1] not in "？！。，":
    output += "。"

print output
o = open("test/result","w")
o.write("Orignial Text:\n"+text+"\n")
o.write("Output:\n"+output+"\n")
