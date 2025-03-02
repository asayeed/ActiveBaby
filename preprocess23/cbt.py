from smart_open import open
from normalize import clean



def preprocess(f):
    for line in f:
        line = line.strip()

        line = line.replace("-LRB-", "(")
        line = line.replace("-LCB-", "{")
        line = line.replace("-LSB-", "[")
        line = line.replace("-RRB-", ")")
        line = line.replace("-RCB-", "}")
        line = line.replace("-RSB-", "]")

        line = line.replace("`` ", '"')
        line = line.replace("``", '"')
        line = line.replace(" ''", '"')
        line = line.replace("''", '"')

        if len(line) == 0:
            yield ""
            continue

        line = clean(line)

        yield line


with open("/srv/data/xloish/babyLM2023/babylm_10M/cbt.train") as f:
    with open("../data/processed23/cbt.txt", 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
