# conding: utf-8
"""Beam pipeline convert TF Record to image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app, flags
import apache_beam as beam
from apache_beam import Pipeline
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
import cv2
import tensorflow as tf

# It's necessary because runtime error raise in apache beam.
tf.enable_eager_execution()

import transform

features = {
    'image': tf.FixedLenFeature((), tf.string, default_value=''),
    'label': tf.FixedLenFeature((), tf.int64, default_value=0)
}


class MyParser(transform.ParseTFRecord):

  def __init__(self, features=None, shape=None, dtype=tf.float32, label=None):
    super(MyParser, self).__init__(
        features=features, shape=shape, dtype=dtype, label=label)

  def decode_tensor(self, tensor_dict):
    image = tensor_dict['image']
    image = tf.decode_raw(image, self._dtype)
    image *= 255.0
    image = tf.cast(image, tf.uint8)
    image = tf.reshape(image, self._shape)
    tensor_dict['image'] = image
    return tensor_dict


class MySaver(transform.WriteImage):

  def __init__(self, dest=None, label=None):
    super(MySaver, self).__init__(dest=dest, label=label)

  def create_fileame(self, label):
    if label == 1:
      return 'ok-{}.png'.format(time.time())
    return 'ng-{}.png'.format(time.time())

  def save(self, tensor_dict):
    image, label = tensor_dict['image'], tensor_dict['label']
    image = image.numpy()
    filename = os.path.join(self._dest, self.create_fileame(label.numpy()))
    im = Image.fromarray(image)
    im.save(filename)


class BGRParser(transform.ParseTFRecord):

  def __init__(self, features=None, shape=None, dtype=tf.float32, label=None):
    super(BGRParser, self).__init__(
        features=features, shape=shape, dtype=dtype, label=label)

  def decode_tensor(self, tensor_dict):
    image = tensor_dict['image']
    image = tf.decode_raw(image, self._dtype)
    image = tf.reshape(image, self._shape)
    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_dict['image'] = image
    label = tensor_dict['label'].numpy()
    tensor_dict['label'] = label
    return tensor_dict


class WriteTFRecord(beam.PTransform):

  def __init__(self, dest=None, label=None):
    super(WriteTFRecord, self).__init__(label=label)
    self._dest = dest

  def make_example(self, tensor_dict):
    """Make example proto has features of image and label.
    
    Args:
      image: Image bytes.
      label: Label.

    Returns:
      Exmaple proto string.
    """
    image = tensor_dict['image']
    image = image.tobytes()
    label = tensor_dict['label']
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label':
                tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            })).SerializeToString()

  def expand(self, pcoll):
    return (pcoll | 'make example' >> beam.Map(self.make_example) | 'write tfrecord'
            >> beam.io.WriteToTFRecord(os.path.join(self._dest, 'ok'), file_name_suffix='.tfrecord'))


def main(_):
  try:
    os.makedirs(FLAGS.dest)
  except IOError:
    pass
  options = PipelineOptions()
  options.view_as(StandardOptions).runner = 'DirectRunner'
  with beam.Pipeline(options=options) as p:
    (p | 'read tfrecord by pattern' >> beam.io.ReadFromTFRecord(FLAGS.pattern) |
     'process image' >> MyParser(features=features, shape=FLAGS.shape) |
     'save image' >> MySaver(dest=FLAGS.dest))


def bgrmain(_):
  try:
    os.makedirs(FLAGS.dest)
  except IOError:
    pass
  options = PipelineOptions()
  options.view_as(StandardOptions).runner = 'DirectRunner'
  with beam.Pipeline(options=options) as p:
    (p | 'read tfrecord by pattern' >> beam.io.ReadFromTFRecord(FLAGS.pattern) |
     'process image' >> BGRParser(features=features, shape=FLAGS.shape) |
     'write tfrecord' >> WriteTFRecord(FLAGS.dest))


def define_flags():
  flags.DEFINE_string(
      name='pattern', help='TF Rercord file name pattren', default='*.tfrecord')
  flags.DEFINE_string(
      name='dest', help='Destination direcoty of parsed image', default=None)
  flags.DEFINE_list(name='shape', help='Shape of image', default=[224, 224, 3])


if __name__ == '__main__':
  define_flags()
  FLAGS = flags.FLAGS
  app.run(bgrmain)
