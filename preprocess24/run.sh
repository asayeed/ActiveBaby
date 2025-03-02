mkdir -p ../data/processed24

python3 childes.py
python3 bnc_spoken.py
python3 gutenberg.py
python3 open_subtitles.py
python3 simple_wiki.py
python3 switchboard.py

cat ../data/processed24/childes.txt ../data/processed24/bnc_spoken.txt ../data/processed24/gutenberg.txt ../data/processed24/open_subtitles.txt ../data/processed24/simple_wiki.txt ../data/processed24/switchboard.txt  > ../data/processed24/all.txt

python3 segment.py
