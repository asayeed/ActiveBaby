
# UPDATE: SEE THE dev2024 BRANCH FOR THE LATEST BABYLM (2024) SUBMISSION 

# README

# ActiveBaby

Repo for BabyLM challenge

## step by step process

### BASELINE

1. RoBERTa over entire corpus.

2. Longformer over entire corpus.

3. Evaluation metrics.

### FANCIER

1. RoBERTa over entire corpus.

2. Trigram (?) surprisals over entire corpus.

2.1 Construct/rescale surprisals into surprisal vectors (SurprisalSpace)

- - via some external HMM tool (like hmmlearn)

3. Split into Initial and Pool

4. Longformer over Initial

5. Per-sentence perplexity over Initial.

6. top-k perplexity sentences (Centroids)

7. kNN in SurprisalSpace of Pool -> add to Initial

8. Goto 2.

9. Evaluation metrics.


### Monitor of training process
https://wandb.ai/tony-xudong-hong/huggingface/runs/v9l8f6mg/overview?workspace=user-tony-xudong-hong


## Timeline

~~January 2023: Training data released~~

~~March 2023: Shared evaluation pipeline released~~

**July 15, 2023:** Models and results due

**August 1, 2023:** Paper submissions due

**Date TBA:** Shared task presented at CoNLL
