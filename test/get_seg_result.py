# -*- coding:utf8 -*-

f = open("test/tmp/input.list")

lines = f.readlines()
result = ""
for line in lines:
  line = line.split()
  if len(line) >= 1:
    word = line[0]
    result += word + " "

print result 
