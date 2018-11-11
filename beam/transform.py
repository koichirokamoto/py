# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import os

import apache_beam as beam
from PIL import Image
import tensorflow as tf


class ParseTFRecord(beam.PTransform):

  def __init__(self, features=None, shape=None, dtype=tf.float32, label=None):
    super(ParseTFRecord, self).__init__(label=label)
    self._features = features
    self._shape = shape
    self._dtype = dtype

  def decode_tensor(self, tensor_dict):
    raise NotImplementedError

  def parse(self, example_proto):
    return tf.parse_single_example(example_proto, features=self._features)

  def expand(self, pcoll):
    return (pcoll | 'parse example protobuf' >> beam.Map(self.parse) |
            'reshape image' >> beam.Map(self.decode_tensor))


class WriteImage(beam.PTransform):

  def __init__(self, dest=None, label=None):
    super(WriteImage, self).__init__(label=label)
    self._dest = dest

  def create_fileame(self, label):
    raise NotImplementedError

  def save(self, tensor_dict):
    image, label = tensor_dict['image'], tensor_dict['label']
    image = image.numpy()
    filename = os.path.join(self._dest, self.create_fileame(label.numpy()))
    im = Image.fromarray(image)
    im.save(filename)

  def expand(self, pcoll):
    return (pcoll | 'write image' >> beam.Map(self.save))
