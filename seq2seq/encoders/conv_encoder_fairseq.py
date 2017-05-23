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
An encoder that conv over embeddings, as described in
https://arxiv.org/abs/1705.03122.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pydoc import locate

import tensorflow as tf

from seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.encoders.pooling_encoder import _create_position_embedding


class ConvEncoderFairseq(Encoder):
  """A deep convolutional encoder, as described in
  https://arxiv.org/abs/1705.03122. The encoder supports optional positions
  embeddings.

  Params:
    attention_cnn.units: Number of units in `cnn_a`. Same in each layer.
    attention_cnn.kernel_size: Kernel size for `cnn_a`.
    attention_cnn.layers: Number of layers in `cnn_a`.
    embedding_dropout_keep_prob: Dropout keep probability
      applied to the embeddings.
    output_cnn.units: Number of units in `cnn_c`. Same in each layer.
    output_cnn.kernel_size: Kernel size for `cnn_c`.
    output_cnn.layers: Number of layers in `cnn_c`.
    position_embeddings.enable: If true, add position embeddings to the
      inputs before pooling.
    position_embeddings.combiner_fn: Function used to combine the
      position embeddings with the inputs. For example, `tensorflow.add`.
    position_embeddings.num_positions: Size of the position embedding matrix.
      This should be set to the maximum sequence length of the inputs.
  """

  def __init__(self, params, mode, name="conv_encoder"):
    super(ConvEncoderFairseq, self).__init__(params, mode, name)
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])

  @staticmethod
  def default_params():
    return {
        "cnn.layers": 4,
        "cnn.nhids": "512,512,512,512",
        "cnn.kwidths": "3,3,3,3",
        "cnn.nhid_default": 256,
        "cnn.kwidth_default": 3,
        "embedding_dropout_keep_prob": 0.8,
        "nhid_dropout_keep_prob": 0.8,
        "word_embeddings.size": 512,
        "position_embeddings.enable": True,
        "position_embeddings.combiner_fn": "tensorflow.add",
        "position_embeddings.num_positions": 100,
    }
   
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
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)
    

  def encode(self, inputs, sequence_length):
    
    embed_size = inputs.get_shape().as_list()[-1]
    if self.params["position_embeddings.enable"]:
      positions_embed = _create_position_embedding(
          embedding_dim=inputs.get_shape().as_list()[-1],
          num_positions=self.params["position_embeddings.num_positions"],
          lengths=sequence_length,
          maxlen=tf.shape(inputs)[1])
      inputs = self._combiner_fn(inputs, positions_embed)
    
    
    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=inputs,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)
    
    with tf.variable_scope("encoder_cnn"):    
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
          next_layer = tf.contrib.layers.conv2d(
            inputs=next_layer,
            num_outputs=nout*2,
            kernel_size=kwidths_list[layer_idx],
            padding="SAME",   #should take attention
            activation_fn=None)
          next_layer = self.gated_linear_units(next_layer)
          next_layer += res_inputs
        next_layer = self.linear_mapping(next_layer, embed_size, var_scope_name="linear_mapping_after_cnn")
      
      cnn_c_output = next_layer + inputs 
            

    final_state = tf.reduce_mean(cnn_c_output, 1)

    return EncoderOutput(
        outputs=next_layer,
        final_state=final_state,
        attention_values=cnn_c_output,
        attention_values_length=sequence_length)
