# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Definition of a basic seq2seq model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate
import tensorflow as tf
from seq2seq.contrib.seq2seq import helper as tf_decode_helper

from seq2seq.models.seq2seq_model import Seq2SeqModel
from seq2seq.graph_utils import templatemethod
from seq2seq.models import bridges
from seq2seq.infer import beam_search

class ConvSeq2Seq(Seq2SeqModel):
  """Basic Sequence2Sequence model with a unidirectional encoder and decoder.
  The last encoder state is used to initialize the decoder and thus both
  must share the same type of RNN cell.

  Args:
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self, params, mode, name="conv_seq2seq"):
    super(ConvSeq2Seq, self).__init__(params, mode, name)
    self.encoder_class = locate(self.params["encoder.class"])
    self.decoder_class = locate(self.params["decoder.class"])

  @staticmethod
  def default_params():
    params = Seq2SeqModel.default_params().copy()
    params.update({
        "attention.class": "AttentionLayerFairseq",
        "attention.params": {}, # Arbitrary attention layer parameters
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "encoder.class": "seq2seq.encoders.ConvEncoderFairseq",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.ConvDecoderFairseq",
        "decoder.params": {}  # Arbitrary parameters for the decoder
    })
    return params
  
  def source_embedding_fairseq(self):
    """Returns the embedding used for the source sequence.
    """
    return tf.get_variable(
        name="W",
        shape=[self.source_vocab_info.total_size, self.params["embedding.dim"]],
        initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=0.1))

  def target_embedding_fairseq(self):
    """Returns the embedding used for the target sequence.
    """
    if self.params["embedding.share"]:
      return self.source_embedding_fairseq()
    return tf.get_variable(
        name="W",
        shape=[self.target_vocab_info.total_size, self.params["embedding.dim"]],
        initializer=tf.random_normal_initializer(
            mean=0.0,
            stddev=0.1))


  def _create_decoder(self, encoder_output, features, _labels):
    attention_class = locate(self.params["attention.class"]) or \
      getattr(decoders.attention, self.params["attention.class"])
    attention_layer = attention_class(
        params=self.params["attention.params"], mode=self.mode)

    # If the input sequence is reversed we also need to reverse
    # the attention scores.
    reverse_scores_lengths = None
    if self.params["source.reverse"]:
      reverse_scores_lengths = features["source_len"]
      if self.use_beam_search:
        reverse_scores_lengths = tf.tile(
            input=reverse_scores_lengths,
            multiples=[self.params["inference.beam_search.beam_width"]])

    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size,
        attention_values=encoder_output.attention_values,
        attention_values_length=encoder_output.attention_values_length,
        attention_keys=encoder_output.outputs,
        attention_fn=attention_layer,
        reverse_scores_lengths=reverse_scores_lengths)

  def _decode_train(self, decoder, _encoder_output, _features, labels):
    """Runs decoding in training mode"""
    target_embedded = tf.nn.embedding_lookup(self.target_embedding_fairseq(),
                                             labels["target_ids"])

    return decoder(_encoder_output, labels=target_embedded[:,:-1], sequence_length=labels["target_len"]-1)

  def _decode_infer(self, decoder, _encoder_output, features, labels):
    """Runs decoding in inference mode"""
    config = beam_search.BeamSearchConfig(
        beam_width=self.params["inference.beam_search.beam_width"],
        vocab_size=self.target_vocab_info.total_size,
        eos_token=self.target_vocab_info.special_vocab.SEQUENCE_END,
        length_penalty_weight=self.params[
            "inference.beam_search.length_penalty_weight"],
        choose_successors_fn=getattr(
            beam_search,
            self.params["inference.beam_search.choose_successors_fn"]))

    target_start_id = self.target_vocab_info.special_vocab.SEQUENCE_START
    start_tokens=tf.fill([batch_size], target_start_id)
    
    return decoder(_encoder_output, target_embedding=self.target_embedding_fairseq(), start_tokens=start_tokens, config=config)

  @templatemethod("encode")
  def encode(self, features, labels):
    source_embedded = tf.nn.embedding_lookup(self.source_embedding_fairseq(),
                                             features["source_ids"])
    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    print('eval_feature_shape', source_embedded.get_shape().as_list())
    print('eval_label_shape', labels["target_ids"].get_shape().as_list())
    return encoder_fn(source_embedded, features["source_len"])

  @templatemethod("decode")
  def decode(self, encoder_output, features, labels):
    
    decoder = self._create_decoder(encoder_output, features, labels)
    if self.use_beam_search:
      decoder = self._get_beam_search_decoder(decoder)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      return self._decode_infer(decoder, encoder_output, features,
                                labels)
    else:
      return self._decode_train(decoder, encoder_output, features,
                                labels)
