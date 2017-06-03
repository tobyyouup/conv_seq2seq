export DATA_PATH=/datadrive/xutan/nmt/seq2seq/data/wmt16_en_de

export VOCAB_SOURCE=${DATA_PATH}/vocab.bpe.32000
export VOCAB_TARGET=${DATA_PATH}/vocab.bpe.32000
export TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.bpe.32000.en
export TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.bpe.32000.de
export DEV_SOURCES=${DATA_PATH}/newstest2013.tok.bpe.32000.en
export DEV_TARGETS=${DATA_PATH}/newstest2013.tok.bpe.32000.de

export DEV_TARGETS_REF=${DATA_PATH}/newstest2013.tok.de
export TRAIN_STEPS=1000000


export MODEL_DIR=/dev/nmt/nmt_tutorial
mkdir -p $MODEL_DIR


python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR


'''

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${PRED_DIR}/predictions.txt

'''
