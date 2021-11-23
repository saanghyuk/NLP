

- `git clone https://github.com/kh-kim/subword-nmt.git`





- How to do 

  1. `./clone.sh`

  2. `cut -f2 review.sorted.uniq.refined.tsv > review.sorted.uniq.refined.tsv.text`

  3. `cat ./review.sorted.uniq.refined.tsv.text | mecab -O wakati | python post_tokenize.py ./review.sorted.uniq.refined.tsv.text > review.sorted.uniq.refined.tsv.text.tok`

     이게 뭐한거냐면, 원래 띄어쓰기가 있던 부분에 _를 붙인 것. 지금 형태소 별로 단어가 떨어져있으니깐, 헷갈림. 즉, 원래 띄어쓰기가 있던 부분과 mecab을 통해 띄어쓰기가 붙은 부분을 구분 가능. 

  4. `python ./subword-nmt/learn_bpe.py -h`

     python ./subword-nmt/learn_bpe.py --input 

     `python3 ./subword-nmt/learn_bpe.py --input ./review.sorted.uniq.refined.tsv.text.tok --output ./model --symbols 30000`

  5. 적용
     - `python ./subword-nmt/apply_bpe.py --codes ./model < review.sorted.uniq.refined2.tsv.text.tok > review.sorted.uniq.refined.tsv.text.tok.bpe`
  6. 파일 원래대로 합쳐주기
     - `cut -f1 review.sorted.uniq.refined.tsv > review.sorted.uniq.refined.tsv.label`

  7. Shell Script
     - 

