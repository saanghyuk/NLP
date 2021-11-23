import sys
import re

# python refine.py refine.regex.txt 1 < review.sorted.uniq.tsv > review.sorted.uniq.refined.tsv
# ^(positive|negative)\t[,.0-9\-=|;]+$


def read_regex(fn):
    regexs = []

    f = open(fn, 'r')

    for line in f:
        if not line.startswith("#"):
            tokens = line.split('\t')

            if len(tokens) == 1:
                tokens += [' ']
            tokens[0] = tokens[0][:-
                                  1] if tokens[0].endswith('\n') else tokens[0]
            tokens[1] = tokens[1][:
                                  -1] if tokens[1].endswith('\n') else tokens[1]
            regexs += [(tokens[0], tokens[1])]

    f.close()

    return regexs


if __name__ == "__main__":
    # input file
    fn = sys.argv[1]
    # if assuming file is tsv file, which index you want to check?
    target_index = int(sys.argv[2])
    regexs = read_regex(fn)
    index = 1
    for line in sys.stdin:

        if line.strip() != "":
            columns = line.strip().split('\t')
            for r in regexs:
                try:
                    columns[target_index] = re.sub(
                        r'%s' % r[0], r[1], columns[target_index].strip())
                except:
                    pass

            sys.stdout.write('\t'.join(columns) + "\n")
            index += 1
        else:
            sys.stdout.write('\n')
