import sys, fileinput
from mosestokenizer import *

if __name__ == "__main__":
    with MosesTokenizer('en') as tokenize:
        for line in fileinput.input():
            if line.strip() != "":
                tokens = tokenize(line.strip())

                sys.stdout.write(" ".join(tokens) + "\n")
            else:
                sys.stdout.write('\n')
