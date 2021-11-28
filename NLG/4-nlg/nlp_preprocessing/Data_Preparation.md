

- **Check the number of lines of data**

  - `wc -l ./*.txt`
  - `wc ./*.txt`

- **Preprocessing**

  1. Check the format of data
     - `head ./raw/1_구어체\(1\).txt `

  2. Merge
     - ` cat ./raw/*.txt > corpus.tsv`
     - `wc -l ./corpus.tsv`

  3. Split Train/Test/Valid Data Set

     - Shuffle - `shuf ./corpus.tsv > ./corpus.shuf.tsv`

     - `head -n 1200000 ./corpus.shuf.tsv > ./corpus.shuf.train.tsv`
     - ` tail -n 402418 ./corpus.shuf.tsv | head -n 200000 > ./corpus.shuf.valid.tsv`

  4. Tokenization

     - Korean and English should be tokenized repectively
     - `cut
        -f1 ./corpus.shuf.train.tsv > ./corpus.shuf.train.ko ; cut -f2 ./corpus.shuf.train.tsv > ./corpus.shuf.tra
       in.en`

     - `head -5 ./corpus.shuf.train.*`
     - `cut
        -f1 ./corpus.shuf.valid.tsv > ./corpus.shuf.valid.ko ; cut -f2 ./corpus.shuf.valid.tsv > ./corpus.shuf.val
       id.en`
     - `cut -f1 ./corpus.shuf.test.tsv > ./corpus.shuf.test.ko ; cut -f2 ./corpus.shuf.test.tsv > ./corpus.shuf.test.e
       n`

-  **Tokenize**

  - Sample : `head -n
    5 ./data/corpus.shuf.train.ko| mecab -O wakati`
    - But with this, it's hard to revert to the original text later. 
  - **KOR** : `cat ./data/corpus.shuf.test.ko| mecab -O wakati -b 99999 | python ./post_tokenize.py ./data/corpus.shuf.test.ko > ./data/corpus.shuf.test.tok.ko`
    - How to revert this file to original?
      - `head -n 5 ./data/corpus.shuf.test.tok.ko | python detokenizer.py`

  - **ENG** :  `cat ./data/corpus.shuf.test.en | python ./tokenizer.py | python ./post_tokenize.py ./data/corpus.shuf.test.en > ./data/corpus.shuf.test.tok.en`

  

  - `cat ./data/corpus.shuf.train.ko | mecab -O wakati | pyt
    hon post_tokenize.py ./data/corpus.shuf.train.ko > ./data/corpus.shuf.train.tok.ko`

  - ` cat ./data/corpus.shuf.train.en | python tokenizer.py| python post_tokenize.py ./data/corpus.shuf.train.en > ./data/corpus.shuf.train.tok.en`

  - `cat ./data/corpus.shuf.valid.ko | mecab -O wakati -b 99999 | python post_tokenize.py ./data/corpus.shuf.valid.ko > ./data/corpus.shuf.valid.tok.ko &`
  - `cat ./data/corpus.shuf.valid.en | python tokenizer.py | python post_tokenize.py ./data/corpus.shuf.valid.en > ./data/corpus.shuf.valid.tok.en`

- **Subword Segmentation**

  - One Underbar :  Tokenized in the tokenzied stage. 

    Two Underbar : This means blank from the original document. 

    Blank : Divide in this subword segmentation stage. 

  - `git clone git@github.com:kh-kim/subword-nmt.git`

  - **Learn**

    -  `python ./subword-nmt/learn_bpe.py --input ./data/corpus.shuf.train.tok.en --output bpe.en.model --symbols 50000 --verbose`
    - `head bpe.en.model` # this is merge instruction
    - `python ./subword-nmt/learn_bpe.py --input ./data/corpus.shuf.train.tok.ko --output bpe.ko.model --symbols 30000 --verbose`

  - **Apply**
    - `cat ./data/corpus.shuf.train.tok.ko | python subword-nmt/apply_bpe.py -c ./bpe.ko.model > ./data/corpus.shuf.train.tok.bpe.ko ; cat ./data/corpus.shuf.valid.tok.ko | python subword-nmt/apply_bpe.py -c ./bpe.ko.model > ./data/corpus.shuf.valid.tok.bpe.ko ; cat ./data/corpus.shuf.test.tok.ko | python subword-nmt/apply_bpe.py -c ./bpe.ko.model > ./data/corpus.shuf.test.tok.bpe.ko`
    - `cat ./data/corpus.shuf.train.tok.en | python subword-nmt/apply_bpe.py -c ./bpe.en.model > ./data/corpus.shuf.train.tok.bpe.en ; cat ./data/corpus.shuf.valid.tok.en | python subword-nmt/apply_bpe.py -c ./bpe.en.model > ./data/corpus.shuf.valid.tok.bpe.en ; cat ./data/corpus.shuf.test.tok.en | python subword-nmt/apply_bpe.py -c ./bpe.en.model > ./data/corpus.shuf.test.tok.bpe.en`

    