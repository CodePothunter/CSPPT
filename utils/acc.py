import sys

file = open(sys.argv[1])

def get_chunks(labels):
    chunks = []
    start_idx,end_idx = 0,0
    for idx in range(1,len(labels)-1):
        chunkStart, chunkEnd = False,False
        if labels[idx-1] != 'O':
            prevTag, prevType = labels[idx-1].split('-')
        else:
            prevTag, prevType = 'O', 'O'
        if labels[idx] != 'O':
            Tag, Type = labels[idx].split('-')
        else:
            Tag, Type = 'O', 'O'
        if labels[idx+1] != 'O':
            nextTag, nextType = labels[idx+1].split('-')
        else:
            nextTag, nextType = 'O', 'O'

        if Tag == 'B' or (prevTag == 'O' and Tag == 'I'):
            chunkStart = True
        if Tag != 'O' and prevType != Type:
            chunkStart = True

        if Tag in ('B','I') and nextTag in ('B','O'):
            chunkEnd = True
        if Tag != 'O' and Type != nextType:
            chunkEnd = True

        if chunkStart:
            start_idx = idx
        if chunkEnd:
            end_idx = idx
            chunks.append((start_idx,end_idx,Type))
            start_idx,end_idx = 0,0
    return chunks

TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
labels = []
preds = []
for line in file:
    line = line.strip()
    if line != "":
        word, label, pred = line.split(' ')
        labels.append(label)
        preds.append(pred)
        if line == "EOS O O":
            label_chunks = get_chunks(labels)
            pred_chunks = get_chunks(preds)
            for pred_chunk in pred_chunks:
                if pred_chunk in label_chunks:
                    TP += 1
                else:
                    FP += 1
            for label_chunk in label_chunks:
                if label_chunk not in pred_chunks:
                    FN += 1
            labels, preds = [], []
print TP/(TP+FP), TP/(TP+FN), 2*TP/(2*TP+FN+FP)
