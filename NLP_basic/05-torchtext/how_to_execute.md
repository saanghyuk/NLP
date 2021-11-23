

- `wc -l ./review.sorted.uniq.refined.tok.*`

  - 각 몇개 있는지 확인. 

- Train/Test

  - Shuffling : `gshuf < ./review.sorted.uniq.refined.tok.tsv > review.sorted.uniq.refined.tok.shuf.tsv`

  - `head -n 252680 ./review.sorted.uniq.refined.tok.shuf.tsv > review.sorted.uniq.refined.tok.train.tsv`

  - `tail -n 50000 ./review.sorted.uniq.refined.tok.shuf.tsv > review.sorted.uniq.refined.tok.test.tsv`
  - 

  