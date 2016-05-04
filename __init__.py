# -*- coding:utf8 -*- 
import sys
from subprocess import Popen, PIPE
args = ["bash", "CSPPT/test.sh"]


def getline(lines):
    words = []
    labels = []
    for line in lines:
        line = line.split()
        words.append(line[0])
        labels.append(line[-1])
    return words, labels
def get_result():
    i1 = open("CSPPT/test/tmp/input.list", "r")
    i2 = open("CSPPT/test/tmp/input.list.result", "r")
    i = open("CSPPT/test/tmp/input", "r")
    
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
    o = open("CSPPT/test/result","w")
    o.write("Orignial Text:\n"+text+"\n")
    o.write("Output:\n"+output+"\n")
    return output

def get_punc(text):
    error = 1
    while error != "":
        p = Popen(args, stdin=PIPE, stderr=PIPE)
        _, error = p.communicate(text)
        print error
    result = get_result()  
    return result.decode("utf-8", 'ignore')



