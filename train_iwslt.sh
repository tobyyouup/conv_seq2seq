export DATA_PATH=/home/xutan/nmt/seq2seq/data/iwslt14.tokenized.de-en

export VOCAB_SOURCE=${DATA_PATH}/vocab.de
export VOCAB_TARGET=${DATA_PATH}/vocab.en
export TRAIN_SOURCES=${DATA_PATH}/train.de
export TRAIN_TARGETS=${DATA_PATH}/train.en
export DEV_SOURCES=${DATA_PATH}/valid.de.full
export DEV_TARGETS=${DATA_PATH}/valid.en.full
export TEST_SOURCES=${DATA_PATH}/test.de
export TEST_TARGETS=${DATA_PATH}/test.en

export TRAIN_STEPS=1000000


export MODEL_DIR=/dev/nmt/nmt_tutorial_iwslt_decay
mkdir -p $MODEL_DIR


'''
python -m bin.train \
  --config_paths="
      ./example_configs/iwslt.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --eval_every_n_steps 5000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
'''


export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}
#DEV_SOURCES=test.en
python -m bin.infer \
  --tasks "
    - class: DecodeText
#    - class: DumpBeams
#      params:
#        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt

