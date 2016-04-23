
import sys,random
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s", "--seed", action="store",
                  dest="seed",
                  default=1000,
                  help="random seed")
(options, args) = parser.parse_args()
seed = int(options.seed)

if len(args) != 1:
  parser.usage = "%prog [options] <data-file>"
  parser.print_help()
  sys.exit(0)

data_file = open(args[0],'rb')
line_list = []
for line in data_file:
  line_list.append(line.strip('\n'))
length = len(line_list)
random.seed(seed)

idx_list = range(length)
random.shuffle(idx_list)

for idx in idx_list:
  print line_list[idx]
  #sys.stdout.write(line_list[idx]+'\n')
