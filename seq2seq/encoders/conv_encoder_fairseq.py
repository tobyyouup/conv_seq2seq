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
#from seq2seq.encoders.pooling_encoder import _create_position_embedding
from seq2seq.encoders.conv_encoder_utils import *


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

  def __init__(self, params, mode, pos_embed, name="conv_encoder"):
    super(ConvEncoderFairseq, self).__init__(params, mode, name)
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])
    self.pos_embed = pos_embed 
  @staticmethod
  def default_params():
    return {
        "cnn.layers": 4,
        "cnn.nhids": "256,256,256,256",
        "cnn.kwidths": "3,3,3,3",
        "cnn.nhid_default": 256,
        "cnn.kwidth_default": 3,
        "embedding_dropout_keep_prob": 0.9,
        "nhid_dropout_keep_prob": 0.9,
        "position_embeddings.enable": True,
        "position_embeddings.combiner_fn": "tensorflow.add",
    }
   

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
    
    positions_embed = tf.reverse_sequence(positions_embed, lengths, batch_dim=0, seq_dim=1)  # [[1,2,3,4,PAD,PAD,PAD],[2,3,PAD,PAD,PAD,PAD,PAD]]   [4,2]
    positions_embed = tf.reverse(positions_embed,[1])  # --> [[4,3,2,1,PAD,PAD,PAD],[3,2,PAD,PAD,PAD,PAD,PAD]] --> [[PAD,PAD,PAD,1,2,3,4],[PAD,PAD,PAD,PAD,PAD,2,3]]

    return positions_embed
   


  def encode(self, inputs, sequence_length):
    
    embed_size = inputs.get_shape().as_list()[-1]
    
    if self.params["position_embeddings.enable"]:
      positions_embed = self._create_position_embedding(
          lengths=sequence_length,  # tensor, data lengths
          maxlen=tf.shape(inputs)[1])  # max len in this batch
      inputs = self._combiner_fn(inputs, positions_embed)
    
    
    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=inputs,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)
    
    with tf.variable_scope("encoder_cnn"):    
      next_layer = inputs
      if self.params["cnn.layers"] > 0:
        nhids_list = parse_list_or_default(self.params["cnn.nhids"], self.params["cnn.layers"], self.params["cnn.nhid_default"])
        kwidths_list = parse_list_or_default(self.params["cnn.kwidths"], self.params["cnn.layers"], self.params["cnn.kwidth_default"])
        
        # mapping emb dim to hid dim
        next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0], dropout=self.params["embedding_dropout_keep_prob"], var_scope_name="linear_mapping_before_cnn")      
        next_layer = conv_encoder_stack(next_layer, nhids_list, kwidths_list, {'src':self.params["embedding_dropout_keep_prob"], 'hid': self.params["nhid_dropout_keep_prob"]}, mode=self.mode)
        
        next_layer = linear_mapping_weightnorm(next_layer, embed_size, var_scope_name="linear_mapping_after_cnn")
      ## The encoder stack will receive gradients *twice* for each attention pass: dot product and weighted sum.
      ##cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))  
      cnn_c_output = (next_layer + inputs) * tf.sqrt(0.5) 
            

    final_state = tf.reduce_mean(cnn_c_output, 1)

    return EncoderOutput(
        outputs=next_layer,
        final_state=final_state,
        attention_values=cnn_c_output,
        attention_values_length=sequence_length)
