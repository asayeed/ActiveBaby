# README

## 2024-05-21

- Sharid: run the elc-bert on merl
- Xudong: run our system with pre-processing from elc-bert 
- Asad: figure out how to make animacy work 


`heroic bash/R command line "cat *.dev | shuf -n 30000 | while read -r line; do echo $line | wc -w; done | echo "f <- (c(`paste -s -d, - `)); c(mean(f), median(f), sd(f), summary(f))" | r -p"`

mean 8.96 median 5 stdev 17 max 530



## 2024-05-07

- Start with Charpentier & Samuel but with some curriculum learning 
- Reproduce official results (same data, same hyperparameters, etc) 
	- fine-tuning
	- external "easy" test 
- ACLM 
	- reduce vocabulary size
	- sentence vector of size 7 can be smarter
		- optimize the size as hyperparameter
		- change representation to something smarter than just a resize
		- bin the sentences so we have different models for different length classes 
	- update the semantic space, it's not currently re-evaluated after the each set of sentences is taken out





## Timeline

March 30 2024: Training data released
April 30 2024: Evaluation pipeline released
September 13 2024: Results due
September 20 2024: Papers due
October 8 2024: Peer review begins
October 30 2024: Peer review ends, acceptances and leaderboard released
Late 2024: Presentation of workshop at ML/NLP venue 
