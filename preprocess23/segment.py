import nltk


with open("../data/processed23/segmented.txt", "w") as f:
    for line in open("../data/processed23/all.txt"):
        line = line.strip()

        if len(line) == 0:
            f.write('\n')
            continue

        sentences = nltk.sent_tokenize(line)
        sentences = '\n'.join(sentences) 
        f.write(f"{sentences}[PAR]\n")
