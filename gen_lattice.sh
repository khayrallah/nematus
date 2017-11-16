#!/bin/bash


source ~/.bashrc

# set -u

SEARCH_GRAPH_=$1
SEARCH_GRAPH=$(basename $SEARCH_GRAPH_)
OUTPUT_DIR=$2
THRESHOLD=$3
BPE_CODES=$4
BPE_TYPE=$5
ID=$6

#update with path to subword-nmt
subword_nmt=~/subword-nmt

[[ ! -d $OUTPUT_DIR ]] && mkdir -p $OUTPUT_DIR

stem=$OUTPUT_DIR/${SEARCH_GRAPH}.$ID
if [[ ! -s ${stem}.p=${THRESHOLD}.${BPE_TYPE}-bpe.fst.txt ]]; then
  if [[ ! -s ${stem}.fst.txt ]]; then
    # Get the sentence graph from the graph file for the test set.
    # Next, convert the search graph into the FSM text format and write out the sym table
    grep "^$ID " $SEARCH_GRAPH_ \
      | python  /home/hltcoe/mpost/expts/whale17/scripts/searchgraph_to_fst.py --prefix $OUTPUT_DIR/${SEARCH_GRAPH}
  fi

  if [[ -s ${stem}.fst.txt ]]; then
    if [[ ! -s ${stem}.fst ]]; then
      #comile fst
      fstcompile --isymbols=${stem}.keys --osymbols=${stem}.keys ${stem}.fst.txt  >  ${stem}.fst
    fi

    # determinize and minimize graph
    # Apply pruning before det-min
    # Garnish with topsort
    cat ${stem}.fst \
      | fstprune --weight=$THRESHOLD \
      | fstrmepsilon | fstdeterminize | fstminimize \
      | fsttopsort > ${stem}.p=${THRESHOLD}.fst


if [[ ! -s $OUTPUT_DIR/${SEARCH_GRAPH}.${ID}.${BPE_TYPE}-bpe.keys  ]]; then
    #apply bpe to keys
    f=$OUTPUT_DIR/${SEARCH_GRAPH}.${ID}.keys
    awk '{ print $1 }' $f > ${f}_words_TEMP
    awk '{ print $2 }' $f > ${f}_nums_TEMP

    #remove _
    sed 's/|/ /g' ${f}_words_TEMP > ${f}_words_spaces_TEMP

    $subword_nmt/apply_bpe.py -c $BPE_CODES <  ${f}_words_spaces_TEMP  >  ${f}_words.${BPE_TYPE}-bpe_TEMP

    #replace spaces with |
    sed 's/ /|/g' ${f}_words.${BPE_TYPE}-bpe_TEMP > ${f}_words.${BPE_TYPE}-bpe_pipes_TEMP

    paste -d ' ' ${f}_words.${BPE_TYPE}-bpe_pipes_TEMP ${f}_nums_TEMP > $OUTPUT_DIR/${SEARCH_GRAPH}.${ID}.${BPE_TYPE}-bpe.keys

    sed -i  's,<@@|/@@|e@@|os@@|>,<eos>,g'  $OUTPUT_DIR/${SEARCH_GRAPH}.${ID}.${BPE_TYPE}-bpe.keys
fi

    cat ${stem}.p=${THRESHOLD}.fst  \
      | fstprint --isymbols=${stem}.${BPE_TYPE}-bpe.keys --osymbols=${stem}.${BPE_TYPE}-bpe.keys \
        --fst_field_separator=" "> ${stem}.p=${THRESHOLD}.${BPE_TYPE}-bpe.fst.txt

  fi
fi

rm ${stem}*TEMP

