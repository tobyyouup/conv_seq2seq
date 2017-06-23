# Convolutional Seq2Seq

This is a tensorflow implementation of the [convolutional seq2seq model](https://arxiv.org/abs/1705.03122) released by [Facebook Fairseq](https://github.com/facebookresearch/fairseq).

This implementation is based on the framework of [Google seq2seq project](https://github.com/google/seq2seq), which has a detailed [documentation](https://google.github.io/seq2seq/) on how to use this framework. In this conv seq2seq project, I implement the conv encoder, conv decoder, and attention mechanism, as well as other modules needed by the conv seq2seq model, which is not available in the original seq2seq project. 


## Requirement

- Python 2.7.0+
- [Tensorflow](https://github.com/tensorflow/tensorflow) 1.0+ (this version is strictly required)
- and their dependencies

Please follow [seq2seq project](https://google.github.io/seq2seq/) on how to install the Convolutional Sequence to Sequence Learning project. 
## How to use
For dataset, please follow [seq2seq nmt guides](https://google.github.io/seq2seq/nmt/) to prepare your dataset

The following is an example of how to run iwslt de-en translation task.
### Train
```
export PYTHONIOENCODING=UTF-8
export DATA_PATH="your iwslt de-en data path"

export VOCAB_SOURCE=${DATA_PATH}/vocab.de
export VOCAB_TARGET=${DATA_PATH}/vocab.en
export TRAIN_SOURCES=${DATA_PATH}/train.de
export TRAIN_TARGETS=${DATA_PATH}/train.en
export DEV_SOURCES=${DATA_PATH}/valid.de
export DEV_TARGETS=${DATA_PATH}/valid.en
export TEST_SOURCES=${DATA_PATH}/test.de
export TEST_TARGETS=${DATA_PATH}/test.en

export TRAIN_STEPS=1000000

export MODEL_DIR=${TMPDIR:-/tmp}/nmt_conv_seq2seq
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq.yml,
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

```

### Test

```
export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}
```

#### decode with greedy search
```
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseq" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt

```

#### decode with beam search
```
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt
```

#### calculate BLEU score
```
./bin/tools/multi-bleu.perl ${TEST_TARGETS} < ${PRED_DIR}/predictions.txt
```


For more detailed instructions, please refer to [seq2seq project](https://google.github.io/seq2seq/).


Issues and contributions are warmly welcome.  


