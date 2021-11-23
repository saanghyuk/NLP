import sys

STR = '▁'
TWO_STR = '▁▁'


def detokenization(line):
    if TWO_STR in line:
        line = line.strip().replace(' ', '').replace(TWO_STR, ' ').replace(STR, '').strip()
    else:
        line = line.strip().replace(' ', '').replace(STR, ' ').strip()

    return line


if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() != "":
            buf = []
            for token in line.strip().split('\t'):
                buf += [detokenization(token)]

            sys.stdout.write('\t'.join(buf) + '\n')
        else:
            sys.stdout.write('\n')
