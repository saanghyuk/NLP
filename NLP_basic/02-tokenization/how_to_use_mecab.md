

- echo "아버지가 방에서 나오신다."  | mecab
- echo "아버지가 방에서 나오신다."  | mecab -O wakati

- echo "애플 컴퓨터의 스티브 잡스"  | mecab -O wakati



- 텝 분리 후 2번째에만 먹여야 함. 

- `cut -f2 ./review.sorted.uniq.refined.tsv| head -n 2`  tab 기준으로 잘라서, 앞에 두줄만 보기. 

- `cut -f2 ./review.sorted.uniq.refined.tsv| mecab -O wakati | head -n 10`

- 첫번째 컬럼(Positive|Negative) 리뷰 따온다.
  - `cut -f1 ./review.sorted.uniq.refined.tsv > review.sorted.uniq.refined.label`
- 두번째 컬럼에 mecab
  - `cut -f2 ./review.sorted.uniq.refined.tsv| mecab -O wakati > review.sorted.uniq.refined.text.tok`

- 위 두 파일 합치기
  - `paste review.sorted.uniq.refined.label review.sorted.uniq.refined.text.tok > review.sorted.uniq.refined.tok.tsv`



- or 
  - ` ./tokenize.sh review.sorted.uniq.refined.tsv review.sorted.uniq.refined.tok.tsv`