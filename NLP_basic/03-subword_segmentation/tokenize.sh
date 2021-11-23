FN=$1
TARGET_FN=$2
N_SYMBOLS=30000

cut -f1 ${FN} > ${FN}.label
cut -f2 ${FN} > ${FN}.text

cat ${FN}.text | mecab -O wakati | python post_tokenize.py ${FN}.text > ${FN}.text.tok

python ./subword-nmt/learn_bpe.py --input ${FN}.text.tok --output ./model --symbols ${N_SYMBOLS}
python ./subword-nmt/apply_bpe.py --codes ./model < ${FN}.text.tok > ${FN}.text.tok.bpe

paste ${FN}.label ${FN}.text.tok.bpe > ${TARGET_FN}
