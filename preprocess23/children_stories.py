from smart_open import open
from normalize import clean



def preprocess(f):
    num_non_blank_lines = 0
    for line in f:
        if len(line.strip()) == 0:
            if num_non_blank_lines > 1:
                yield ""

            num_non_blank_lines = 0
            continue

        if line.startswith("    "):
            line = f"[TAB] {' '.join(line.strip().split())}"
        else:
            line = ' '.join(line.strip().split())

        num_non_blank_lines += 1
        line = clean(line, minimal=True)

        yield line

with open("/srv/data/xloish/babyLM2023/babylm_10M/children_stories.train") as f:
    with open("../data/processed23/children_stories.txt", 'w') as g:
        for line in preprocess(f):
            g.write(f"{line}\n")
