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
Base class for sequence decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
from collections import namedtuple
from pydoc import locate

import six
import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops

from seq2seq.graph_module import GraphModule
from seq2seq.configurable import Configurable
from seq2seq.contrib.seq2seq.decoder import Decoder, dynamic_decode
from seq2seq.contrib.seq2seq.decoder import _transpose_batch_time
from seq2seq.encoders.pooling_encoder import _create_position_embedding, position_encoding
from seq2seq.encoders.conv_encoder_utils import ConvEncoderUtils
from seq2seq.inference import beam_search  
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from seq2seq.encoders.encoder import EncoderOutput

class ConvDecoderOutput(
    #namedtuple("ConvDecoderOutput", ["logits", "predicted_ids", "cell_output", "attention_scores", "attention_context"])):
    namedtuple("ConvDecoderOutput", ["logits", "predicted_ids"])): 
    pass

class FinalBeamDecoderOutput(
    namedtuple("FinalBeamDecoderOutput",
               ["predicted_ids", "beam_search_output"])):
    pass

class BeamDecoderOutput(
    namedtuple("BeamDecoderOutput", [
        "logits", "predicted_ids", "log_probs", "scores", "beam_parent_ids"
    ])):
    pass

@six.add_metaclass(abc.ABCMeta)
class ConvDecoder(Decoder, GraphModule, Configurable):
  """An RNN Decoder that uses attention over an input sequence.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_keys: The sequence used to calculate attention scores.
      A tensor of shape `[B, T, ...]`.
    attention_values: The sequence to attend over.
      A tensor of shape `[B, T, input_dim]`.
    attention_values_length: Sequence length of the attention values.
      An int32 Tensor of shape `[B]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
  """

  def __init__(self,
               params,
               mode,
               vocab_size,
               config,
               target_embedding,
               start_tokens,
               enc_output,
               name="conv_decoder_fairseq"):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)
    
    self.vocab_size = vocab_size
    self.config=config
    self.target_embedding=target_embedding 
    self.start_tokens=start_tokens
    self.enc_output=enc_output  
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])
    self.positions_embed = tf.constant(position_encoding(self.params["position_embeddings.num_positions"], target_embedding.get_shape().as_list()[-1]), name="position_encoding") 
    self.current_inputs = None
    self.conv_utils = ConvEncoderUtils()
  @staticmethod
  def default_params():
    return {
        "cnn.layers": 3,
        "cnn.nhids": "512,512,512",
        "cnn.kwidths": "3,3,3",
        "cnn.nhid_default": 256,
        "cnn.kwidth_default": 3,
        "embedding_dropout_keep_prob": 0.8,
        "nhid_dropout_keep_prob": 0.8,
        "out_dropout_keep_prob": 0.8,
        "word_embeddings.size": 512,
        "position_embeddings.enable": True,
        "position_embeddings.combiner_fn": "tensorflow.add",
        "position_embeddings.num_positions": 100,
        "max_decode_length": 100,
        "init_scale": 0.04,
        "nout_embed": 256,
    }
 
  @property
  def batch_size(self):
    return self.config.beam_width

  @property
  def output_size(self):
    return BeamDecoderOutput(
        logits=self.vocab_size,   # need pay attention
        predicted_ids=tf.TensorShape([]),
        log_probs=tf.TensorShape([]),
        scores=tf.TensorShape([]),
        beam_parent_ids=tf.TensorShape([]))

  @property
  def output_dtype(self):
    return BeamDecoderOutput(
        logits=tf.float32,
        predicted_ids=tf.int32,
        log_probs=tf.float32,
        scores=tf.float32,
        beam_parent_ids=tf.int32)
  
  def initialize(self, name=None):
    
    finished = tf.tile([False], [self.config.beam_width])
    beam_state = beam_search.create_initial_beam_state(self.config)    
    
    start_embed = tf.nn.embedding_lookup(self.target_embedding, self.start_tokens)
    start_embed = tf.expand_dims(start_embed, 1)

    return finished, start_embed, beam_state
  
  def finalize(self, outputs, final_state):
    # Gather according to beam search result
    predicted_ids = beam_search.gather_tree(outputs.predicted_ids,
                                            outputs.beam_parent_ids)

    # We're using a batch size of 1, so we add an extra dimension to
    # convert tensors to [1, beam_width, ...] shape. This way Tensorflow
    # doesn't confuse batch_size with beam_width
    outputs = nest.map_structure(lambda x: tf.expand_dims(x, 1), outputs)

    final_outputs = FinalBeamDecoderOutput(
        predicted_ids=tf.expand_dims(predicted_ids, 1),
        beam_search_output=outputs)

    return final_outputs, final_state
  
  def next_inputs(self, sample_ids,name=None):
    finished = math_ops.equal(sample_ids, self.config.eos_token)
    all_finished = math_ops.reduce_all(finished)
    next_inputs = control_flow_ops.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda:  tf.nn.embedding_lookup(self.target_embedding, tf.tile([self.config.eos_token], [self.config.beam_width])),
        lambda: tf.nn.embedding_lookup(self.target_embedding, sample_ids))
    return all_finished, next_inputs

  def add_position_embedding(self, inputs):
    seq_len = inputs.get_shape().as_list()[1]
    seq_pos_embed = self.positions_embed[0:seq_len,:]   
    seq_pos_embed_batch = tf.tile(seq_pos_embed, [self.config.beam_width,1])
    
    return self._combiner_fn(inputs, seq_pos_embed_batch)

  def step(self, time, inputs, state, name=None):
   
    if self.current_inputs == None:
      self.current_inputs = inputs 
    else:
      self.current_inputs = tf.concat([self.current_inputs, inputs], axis=1)
    
    inputs = self.add_position_embedding(self.current_inputs)
      
    logits = self.infer_conv_block(self.enc_output, inputs)
    print('logits', logits.get_shape().as_list())    
    
    bs_output, beam_state = beam_search.beam_search_step(
        time_=time,
        logits=logits,
        beam_state=state,
        config=self.config)
    print('bs_output.predicted_ids', bs_output.predicted_ids.get_shape().as_list())    

    finished, next_inputs = self.next_inputs(sample_ids=bs_output.predicted_ids)
    next_inputs = tf.reshape(next_inputs, [self.config.beam_width, 1, inputs.get_shape().as_list()[-1]])

    outputs = BeamDecoderOutput(
        logits=tf.zeros([self.config.beam_width, self.config.vocab_size]),
        predicted_ids=bs_output.predicted_ids,
        log_probs=beam_state.log_probs,
        scores=bs_output.scores,
        beam_parent_ids=bs_output.beam_parent_ids)
    return outputs, beam_state, next_inputs, finished


    

  def infer_conv_block(self, enc_output, input_embed):
    # Apply dropout to embeddings
    input_embed = tf.contrib.layers.dropout(
        inputs=input_embed,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.INFER)
     
    next_layer = self.conv_block(enc_output, input_embed, False)
    
    shape = next_layer.get_shape().as_list()  
    logits = tf.reshape(next_layer, [-1,shape[-1]])   
    return logits

  def conv_block(self, enc_output, input_embed, is_train=True):
    #with tf.variable_scope("decoder_cnn"):    
    scope="model/conv_seq2seq/decode/conv_decoder_fairseq/decoder"
    with tf.variable_scope("decoder_cnn"):    
      next_layer = input_embed
      if self.params["cnn.layers"] > 0:
        nhids_list = self.conv_utils.parse_list_or_default(self.params["cnn.nhids"], self.params["cnn.layers"], self.params["cnn.nhid_default"])
        kwidths_list = self.conv_utils.parse_list_or_default(self.params["cnn.kwidths"], self.params["cnn.layers"], self.params["cnn.kwidth_default"])
        
        # mapping emb dim to hid dim
        next_layer = self.conv_utils.linear_mapping(next_layer, nhids_list[0], dropout=self.params["embedding_dropout_keep_prob"], var_scope_name="linear_mapping_before_cnn")      
         
        next_layer = self.conv_utils.conv_decoder_stack(input_embed, enc_output, next_layer, nhids_list, kwidths_list, {'src':0.8, 'hid':0.8}, mode=self.mode)
    
    with tf.variable_scope("softmax"):
      if is_train:
        next_layer = self.conv_utils.linear_mapping(next_layer, self.params["nout_embed"], var_scope_name="linear_mapping_after_cnn")
      else:         
        next_layer = self.conv_utils.linear_mapping(next_layer[:,-1:,:], self.params["nout_embed"], var_scope_name="linear_mapping_after_cnn")
      next_layer = tf.contrib.layers.dropout(
        inputs=next_layer,
        keep_prob=self.params["out_dropout_keep_prob"],
        is_training=is_train)
     
      next_layer = self.conv_utils.linear_mapping(next_layer, self.vocab_size, in_dim=self.params["nout_embed"], dropout=self.params["out_dropout_keep_prob"], var_scope_name="logits_before_softmax")
      
    return next_layer 
 
  def init_params_in_loop(self):
    with tf.variable_scope("decoder"):
      initial_finished, initial_inputs, initial_state = self.initialize()
      logits = self.infer_conv_block(self.enc_output, initial_inputs)
      

  def print_tensor_shape(self, tensor, name):
    print(name, tensor.get_shape().as_list()) 
  
  def conv_decoder_infer(self):
    maximum_iterations = self.params["max_decode_length"]
    
    self.init_params_in_loop()
    tf.get_variable_scope().reuse_variables()    
    outputs, final_state = dynamic_decode(
        decoder=self,
        output_time_major=True,
        impute_finished=False,
        maximum_iterations=maximum_iterations)
 
    return outputs, final_state

  def conv_decoder_train(self, enc_output, labels, sequence_length):
    embed_size = labels.get_shape().as_list()[-1]
    if self.params["position_embeddings.enable"]:
      positions_embed = _create_position_embedding(
          embedding_dim=embed_size,
          num_positions=self.params["position_embeddings.num_positions"],
          lengths=sequence_length,
          maxlen=tf.shape(labels)[1])
      labels = self._combiner_fn(labels, positions_embed)
     
    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=labels,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)
    
    next_layer = self.conv_block(enc_output, inputs, True)
      
       
    logits = _transpose_batch_time(next_layer)   

    sample_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    #sample_ids = math_ops.cast(math_ops.argmax(logits, axis=-1), dtypes.int32)
 
    return ConvDecoderOutput(logits=logits, predicted_ids=sample_ids)

  def _build(self, enc_output, labels=None, sequence_length=None, start_tokens=None):

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      outputs = tf.tile(self.enc_output.outputs, [self.config.beam_width,1,1]) 
      attention_values = tf.tile(self.enc_output.attention_values, [self.config.beam_width,1,1]) 
      self.enc_output = EncoderOutput(
        outputs=outputs,
        final_state=self.enc_output.final_state,
        attention_values=attention_values,
        attention_values_length=self.enc_output.attention_values_length)
      outputs, states = self.conv_decoder_infer()
      return self.finalize(outputs, states)
    else:
      with tf.variable_scope("decoder"):  # when infer, dynamic decode will add decoder scope, so we add here to keep it the same  
        outputs = self.conv_decoder_train(enc_output=enc_output, labels=labels, sequence_length=sequence_length)
        states = None
        return outputs, states
