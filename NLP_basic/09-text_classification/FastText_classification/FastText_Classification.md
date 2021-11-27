# Classification with FastText



- [Classifacation with FastText](https://github.com/facebookresearch/fastText/#text-classification)

- we need to change the text format of our corpus with corresponding to the fasttext. 

  - `head ./review.sorted.uniq.refined.tok.shuf.train.tsv` 
  - `fasttext supervised -input ./review.sorted.uniq.refined.tok.shuf.train.tsv  -output model` failed to run because format not yet arranged. 

- Regex

  ![1](./1.png)

- `fasttext supervised -input ./review.sorted.uniq.refined.tok.shuf.train.tsv  -output model` Run Again. 
- Test
- Predict
  - `cut -f2 ./review.sorted.uniq.refined.tok.shuf.test.tsv > review.sorted.uniq.refined.tok.shuf.test.no_label.txt`
  - `fasttext predict model.bin review.sorted.uniq.refined.tok.shuf.test.no_label.txt| head