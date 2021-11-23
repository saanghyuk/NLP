FN=$1
TARGET_FN=$2

cut -f1 ${FN} > ${FN}.label
cut -f2 ${FN} | mecab -O wakati -b 20000 > ${FN}.text.tok

paste ${FN}.label ${FN}.text.tok > ${TARGET_FN}
