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
#from seq2seq.encoders.pooling_encoder import _create_position_embedding, position_encoding
from seq2seq.encoders.conv_encoder_utils import *
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
class ConvDecoderFairseqBS(Decoder, GraphModule, Configurable):
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
               pos_embedding,
               start_tokens,
               name="conv_decoder_fairseq"):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)
    
    self.vocab_size = vocab_size
    self.config=config
    self.target_embedding=target_embedding 
    self.start_tokens=start_tokens
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])
    self.pos_embed = pos_embedding
    self.current_inputs = None
    self.initial_state = None

  @staticmethod
  def default_params():
    return {
        "cnn.layers": 3,
        "cnn.nhids": "256,256,256",
        "cnn.kwidths": "3,3,3",
        "cnn.nhid_default": 256,
        "cnn.kwidth_default": 3,
        "embedding_dropout_keep_prob": 0.9,
        "nhid_dropout_keep_prob": 0.9,
        "out_dropout_keep_prob": 0.9,
        "position_embeddings.enable": True,
        "position_embeddings.combiner_fn": "tensorflow.add",
        "max_decode_length": 49,
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
  def print_shape(self, name, tensor):
    print(name, tensor.get_shape().as_list()) 
 
  def _setup(self, initial_state, helper=None):
    self.initial_state = initial_state
  
  def initialize(self, name=None):
    
    finished = tf.tile([False], [self.config.beam_width])
    
    start_tokens_batch = tf.fill([self.config.beam_width], self.start_tokens)
    first_inputs = tf.nn.embedding_lookup(self.target_embedding, start_tokens_batch)
    first_inputs = tf.expand_dims(first_inputs, 1)
    zeros_padding = tf.zeros([self.config.beam_width, self.params['max_decode_length']-1, self.target_embedding.get_shape().as_list()[-1]])
    first_inputs = tf.concat([first_inputs, zeros_padding], axis=1)
    beam_state = beam_search.create_initial_beam_state(self.config)    
    
    outputs = tf.tile(self.initial_state.outputs, [self.config.beam_width,1,1]) 
    attention_values = tf.tile(self.initial_state.attention_values, [self.config.beam_width,1,1]) 
    enc_output = EncoderOutput(
        outputs=outputs,
        final_state=self.initial_state.final_state,
        attention_values=attention_values,
        attention_values_length=self.initial_state.attention_values_length)
    
    
    return finished, first_inputs, (enc_output, beam_state)
  
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

  def _create_position_embedding(self, lengths, maxlen):

    # Slice to size of current sequence
    pe_slice = self.pos_embed[2:maxlen+2, :]
    # Replicate encodings for each element in the batch
    batch_size = tf.shape(lengths)[0]
    pe_batch = tf.tile([pe_slice], [batch_size, 1, 1])

    # Mask out positions that are padded
    positions_mask = tf.sequence_mask(
        lengths=lengths, maxlen=maxlen, dtype=tf.float32)
    positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)

    return positions_embed
  
  def add_position_embedding(self, inputs, time):
    seq_pos_embed = self.pos_embed[2:time+1+2,:]  
    seq_pos_embed = tf.expand_dims(seq_pos_embed, axis=0) 
    seq_pos_embed_batch = tf.tile(seq_pos_embed, [self.config.beam_width,1,1])
    
    return self._combiner_fn(inputs, seq_pos_embed_batch)

  def step(self, time, inputs, state, name=None):
   
    cur_inputs = inputs[:,0:time+1,:] 
    zeros_padding = inputs[:,time+2:,:] 
    cur_inputs_pos = self.add_position_embedding(cur_inputs, time)
    
    enc_output, beam_state = state 
    logits = self.infer_conv_block(enc_output, cur_inputs_pos)
    
    bs_output, beam_state = beam_search.beam_search_step(
        time_=time,
        logits=logits,
        beam_state=beam_state,
        config=self.config)

    finished, next_inputs = self.next_inputs(sample_ids=bs_output.predicted_ids)
    next_inputs = tf.reshape(next_inputs, [self.config.beam_width, 1, inputs.get_shape().as_list()[-1]])
    next_inputs = tf.concat([cur_inputs, next_inputs], axis=1)
    next_inputs = tf.concat([next_inputs, zeros_padding], axis=1)
    next_inputs.set_shape([self.config.beam_width, self.params['max_decode_length'], inputs.get_shape().as_list()[-1]])
    outputs = BeamDecoderOutput(
        logits=tf.zeros([self.config.beam_width, self.config.vocab_size]),
        predicted_ids=bs_output.predicted_ids,
        log_probs=beam_state.log_probs,
        scores=bs_output.scores,
        beam_parent_ids=bs_output.beam_parent_ids)
    return outputs, (enc_output,beam_state), next_inputs, finished


    

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
    with tf.variable_scope("decoder_cnn"):    
      next_layer = input_embed
      if self.params["cnn.layers"] > 0:
        nhids_list = parse_list_or_default(self.params["cnn.nhids"], self.params["cnn.layers"], self.params["cnn.nhid_default"])
        kwidths_list = parse_list_or_default(self.params["cnn.kwidths"], self.params["cnn.layers"], self.params["cnn.kwidth_default"])
        
        # mapping emb dim to hid dim
        next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0], dropout=self.params["embedding_dropout_keep_prob"], var_scope_name="linear_mapping_before_cnn")      
         
        next_layer = conv_decoder_stack(input_embed, enc_output, next_layer, nhids_list, kwidths_list, {'src':self.params["embedding_dropout_keep_prob"], 'hid': self.params["nhid_dropout_keep_prob"]}, mode=self.mode)
    
    with tf.variable_scope("softmax"):
      if is_train:
        next_layer = linear_mapping_weightnorm(next_layer, self.params["nout_embed"], var_scope_name="linear_mapping_after_cnn")
      else:         
        next_layer = linear_mapping_weightnorm(next_layer[:,-1:,:], self.params["nout_embed"], var_scope_name="linear_mapping_after_cnn")
      next_layer = tf.contrib.layers.dropout(
        inputs=next_layer,
        keep_prob=self.params["out_dropout_keep_prob"],
        is_training=is_train)
     
      next_layer = linear_mapping_weightnorm(next_layer, self.vocab_size, in_dim=self.params["nout_embed"], dropout=self.params["out_dropout_keep_prob"], var_scope_name="logits_before_softmax")
      
    return next_layer 
 
  def init_params_in_loop(self):
    with tf.variable_scope("decoder"):
      initial_finished, initial_inputs, initial_state = self.initialize()
      enc_output, beam_sate = initial_state
      logits = self.infer_conv_block(enc_output, initial_inputs)
      

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
      positions_embed = self._create_position_embedding(
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
 
    return ConvDecoderOutput(logits=logits, predicted_ids=sample_ids)

  def _build(self, enc_output, labels=None, sequence_length=None):
    
    if not self.initial_state:
      self._setup(initial_state=enc_output)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      outputs, states = self.conv_decoder_infer()
      return self.finalize(outputs, states)
    else:
      with tf.variable_scope("decoder"):  # when infer, dynamic decode will add decoder scope, so we add here to keep it the same  
        outputs = self.conv_decoder_train(enc_output=enc_output, labels=labels, sequence_length=sequence_length)
        states = None
        return outputs, states
