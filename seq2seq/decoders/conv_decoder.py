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
from seq2seq.encoders.pooling_encoder import _create_position_embedding

class ConvDecoderOutput(
    #namedtuple("ConvDecoderOutput", ["logits", "predicted_ids", "cell_output", "attention_scores", "attention_context"])):
    namedtuple("ConvDecoderOutput", ["logits", "predicted_ids"])): 
    pass


@six.add_metaclass(abc.ABCMeta)
class ConvDecoder(GraphModule, Configurable):
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
               attention_keys,
               attention_values,
               attention_values_length,
               attention_fn,
               reverse_scores_lengths=None,
               name="conv_decoder_fairseq"):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)
    
    self.vocab_size = vocab_size
    self.attention_keys = attention_keys
    self.attention_values = attention_values
    self.attention_values_length = attention_values_length
    self.attention_fn = attention_fn
    self.reverse_scores_lengths = reverse_scores_lengths
    
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])

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
        "word_embeddings.size": 512,
        "position_embeddings.enable": True,
        "position_embeddings.combiner_fn": "tensorflow.add",
        "position_embeddings.num_positions": 100,
        "init_scale": 0.04,
        "nout_embed": 256,
    }
  
  def compute_output(self, cell_output):
    """Computes the decoder outputs."""

    # Compute attention
    att_scores, attention_context = self.attention_fn(
        query=cell_output,
        keys=self.attention_keys,
        values=self.attention_values,
        values_length=self.attention_values_length)

    # TODO: Make this a parameter: We may or may not want this.
    # Transform attention context.
    # This makes the softmax smaller and allows us to synthesize information
    # between decoder state and attention context
    # see https://arxiv.org/abs/1508.04025v5
    softmax_input = tf.contrib.layers.fully_connected(
        inputs=tf.concat([cell_output, attention_context], 1),
        num_outputs=self.cell.output_size,
        activation_fn=tf.nn.tanh,
        scope="attention_mix")

    # Softmax computation
    logits = tf.contrib.layers.fully_connected(
        inputs=softmax_input,
        num_outputs=self.vocab_size,
        activation_fn=None,
        scope="logits")

    return softmax_input, logits, att_scores, attention_context

  def parse_list_or_default(self, params_str, number, default_val, delimitor=','):
    param_list = []
    if params_str == "":
      param_list = [default_val] * number
    else:
      param_list = [int(x) for x in params_str.strip().split(delimitor)]
    return param_list
 
  def linear_mapping(self, inputs, out_dim, dropout=0.0, var_scope_name="linear_mapping"):
    with tf.variable_scope(var_scope_name): 
      input_shape_tensor = tf.shape(inputs)
      input_shape = inputs.get_shape().as_list()
      assert len(input_shape) == 3
      
      inputs = tf.reshape(inputs, [-1, input_shape[-1]])    
      linear_mapping_w = tf.get_variable("linear_mapping_w", [input_shape[-1], out_dim], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
      linear_mapping_b = tf.get_variable("linear_mapping_b", [out_dim], initializer=tf.truncated_normal_initializer(stddev=0.1)) 
      output = tf.matmul(inputs, linear_mapping_w) + linear_mapping_b
      print('xxxxx_params', input_shape, out_dim)
      #output = tf.reshape(output, [input_shape[0], -1, out_dim])
      output = tf.reshape(output, [input_shape_tensor[0], -1, out_dim])
    
    return output    
  
  def gated_linear_units(self, inputs):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    input_pass = inputs[:,:,0:int(input_shape[2]/2)]
    input_gate = inputs[:,:,int(input_shape[2]/2):]
    #input_pass = tf.slice(inputs, [0,0,0], [input_shape[0], input_shape[1], int(input_shape[2]/2)])    
    #input_gate = tf.slice(inputs, [0,0,int(input_shape[2]/2)], [input_shape[0], input_shape[1], int(input_shape[2]/2)])    
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)

  def make_attention(self, target_embed, encoder_output, decoder_hidden, layer_idx):
    with tf.variable_scope("attention_layer_" + str(layer_idx)):
      embed_size = target_embed.get_shape().as_list()[-1]      #k
      dec_hidden_proj = self.linear_mapping(decoder_hidden, embed_size, var_scope_name="linear_mapping_att_query")  # M*N1*k1 --> M*N1*k
      dec_rep = dec_hidden_proj + target_embed
   
      encoder_output_a = encoder_output.outputs
      encoder_output_c = encoder_output.attention_values    # M*N2*K
      att_score = tf.matmul(dec_rep, encoder_output_a, transpose_b=True)  #M*N1*K  ** M*N2*K  --> M*N1*N2
      att_score = tf.nn.softmax(att_score)        
    
      att_out = tf.matmul(att_score, encoder_output_c)    #M*N1*N2  ** M*N2*K   --> M*N1*k
       
      att_out = self.linear_mapping(att_out, decoder_hidden.get_shape().as_list()[-1], var_scope_name="linear_mapping_att_out")
    return att_out


  def conv_decoder_train(self, decoder, enc_output, labels, sequence_length):
    """
    if not isinstance(decoder, Decoder):
      raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                      type(decoder))
    """
    if self.params["position_embeddings.enable"]:
      positions_embed = _create_position_embedding(
          embedding_dim=labels.get_shape().as_list()[-1],
          num_positions=self.params["position_embeddings.num_positions"],
          lengths=sequence_length,
          maxlen=tf.shape(labels)[1])
      labels = self._combiner_fn(labels, positions_embed)
     
    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=labels,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)
    
    with tf.variable_scope("decoder_cnn"):    
      next_layer = inputs
      if self.params["cnn.layers"] > 0:
        nhids_list = self.parse_list_or_default(self.params["cnn.nhids"], self.params["cnn.layers"], self.params["cnn.nhid_default"])
        kwidths_list = self.parse_list_or_default(self.params["cnn.kwidths"], self.params["cnn.layers"], self.params["cnn.kwidth_default"])
        
        # mapping emb dim to hid dim
        next_layer = self.linear_mapping(next_layer, nhids_list[0], dropout=self.params["embedding_dropout_keep_prob"], var_scope_name="linear_mapping_before_cnn")      
        
        for layer_idx in range(len(nhids_list)):
          nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx-1]
          nout = nhids_list[layer_idx]
          if nin != nout:
            #mapping for res add
            res_inputs = self.linear_mapping(next_layer, nout, dropout=self.params["nhid_dropout_keep_prob"], var_scope_name="linear_mapping_cnn_" + str(layer_idx))      
          else:
            res_inputs = next_layer
          #dropout before input to conv
          # special process here, first padd then conv, because tf does not suport padding other than SAME and VALID
          next_layer = tf.pad(next_layer, [[0, 0], [kwidths_list[layer_idx]-1, kwidths_list[layer_idx]-1], [0, 0]], "CONSTANT")
          next_layer = tf.contrib.layers.conv2d(
            inputs=next_layer,
            num_outputs=nout*2,
            kernel_size=kwidths_list[layer_idx],
            padding="VALID",   #should take attention, not SAME but VALID
            activation_fn=None)
          
          layer_shape = next_layer.get_shape().as_list()
          assert len(layer_shape) == 3
          # to avoid using future information 
          next_layer = next_layer[:,0:-kwidths_list[layer_idx]+1,:]

          next_layer = self.gated_linear_units(next_layer)
         
          # add attention
          # decoder output -->linear mapping to embed, + target embed,  query decoder output a, softmax --> scores, scores*encoder_output_c-->output,  output--> linear mapping to nhid+  decoder_output -->
          att_out = self.make_attention(inputs, enc_output, next_layer, layer_idx) 
          next_layer += att_out

          # add res connections
          next_layer += res_inputs
    
    with tf.variable_scope("softmax"):
      next_layer = self.linear_mapping(next_layer, self.params["nout_embed"], var_scope_name="linear_mapping_after_cnn")
      next_layer = self.linear_mapping(next_layer, self.vocab_size, var_scope_name="logits_before_softmax")
       
    logits = _transpose_batch_time(next_layer)   

    #sample_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    sample_ids = math_ops.cast(math_ops.argmax(logits, axis=-1), dtypes.int32)
 
    return ConvDecoderOutput(logits=logits, predicted_ids=sample_ids)


  def _build(self, enc_output, labels, sequence_length):

    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))

    maximum_iterations = None
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      maximum_iterations = self.params["max_decode_length"]

    outputs = self.conv_decoder_train(decoder=self, enc_output=enc_output, labels=labels, sequence_length=sequence_length)
    return outputs, outputs
