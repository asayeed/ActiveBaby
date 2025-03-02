mkdir -p ../data/processed23

python3 aochildes.py
python3 bnc_spoken.py
python3 cbt.py
python3 children_stories.py
python3 gutenberg.py
python3 open_subtitles.py
python3 qed.py
python3 simple_wikipedia.py
python3 switchboard.py
python3 wikipedia.py

cat ../data/processed23/aochildes.txt ../data/processed23/bnc_spoken.txt ../data/processed23/cbt.txt ../data/processed23/children_stories.txt ../data/processed23/gutenberg.txt ../data/processed23/open_subtitles.txt ../data/processed23/qed.txt ../data/processed23/simple_wikipedia.txt ../data/processed23/switchboard.txt ../data/processed23/wikipedia.txt > ../data/processed23/all.txt

python3 segment.py
