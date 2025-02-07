import random
import gc

def RandomShuffle(infile, outfile, deleteSchema=False):
    with open(infile, 'r', encoding='utf-8') as fs:
        arr = fs.readlines()
    if not arr[-1].endswith('\n'):
        arr[-1] += '\n'
    if deleteSchema:
        arr = arr[1:]
    random.shuffle(arr)
    with open(outfile, 'w', encoding='utf-8') as fs:
        fs.writelines(arr)  # Use writelines for efficiency
    del arr
    gc.collect()  # Explicitly call garbage collection

def WriteToBuff(buff, line, out):
    BUFF_SIZE = 1000000
    buff.append(line)
    if len(buff) == BUFF_SIZE:
        WriteToDisk(buff, out)

def WriteToDisk(buff, out):
    with open(out, 'a', encoding='utf-8') as fs:
        fs.writelines(buff)  # Use writelines here as well
    buff.clear()
    gc.collect()

def SubDataSet(infile, outfile1, outfile2, rate):
    out1 = []  # Initialize as empty lists
    out2 = []
    with open(infile, 'r', encoding='utf-8') as fs:
        for line in fs:
            if random.random() < rate:
                out1.append(line)  # Append to the list directly
            else:
                out2.append(line)  # Append to the list directly

    WriteToDisk(out1, outfile1)  # Write after the loop
    WriteToDisk(out2, outfile2)  # Write after the loop
    gc.collect()

def CombineFiles(files, out):
    buff = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as fs:
            for line in fs:
                WriteToBuff(buff, line, out)
    WriteToDisk(buff, out)
    gc.collect()
