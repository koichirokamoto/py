# coding: utf-8
"""Beam pipeline convert TF Record to image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import math
import os
import random
import time

from absl import app, flags
import apache_beam as beam
from apache_beam import Pipeline
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
import cv2
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# It's necessary because runtime error raise in apache beam.
tf.enable_eager_execution()

import transform

features = {
    'image': tf.FixedLenFeature((), tf.string, default_value=''),
    'label': tf.FixedLenFeature((), tf.int64, default_value=0)
}

seed = random.seed(int(time.time()))


def create_filename(prefix):
  return '{}-{}-{}.tfrecord'.format(prefix, time.time(), random.random())


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


class CropParser(transform.ParseTFRecord):

  def __init__(self, features=None, shape=None, dtype=tf.float32, label=None):
    super(RotateParser, self).__init__(
        features=features, shape=shape, dtype=dtype, label=label)

  def decode_tensor(self, tensor_dict):
    image = tensor_dict['image']
    image = tf.decode_raw(image, self._dtype)
    image = tf.reshape(image, self._shape)
    image = tf.random_crop(image, [180, 180, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    tensor_dict['image'] = image.numpy()
    tensor_dict['label'] = 0
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

  def __init__(self, prefix='ok', dest=None, label=None):
    super(WriteTFRecord, self).__init__(label=label)
    self._prefix = prefix
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
            }))

  def write_tfrecord(self, example):
    filename = os.path.join(self._dest, create_filename(self._prefix))
    with tf.python_io.TFRecordWriter(filename) as w:
      w.write(example.SerializeToString())

  def expand(self, pcoll):
    return (pcoll | 'make example' >> beam.Map(self.make_example) |
            'write tfrecord' >> beam.Map(self.write_tfrecord))


def main():
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


def bgrmain():
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


def image2tfrecord():

  def mk(filename):
    img = load_img(filename, target_size=(224, 224))
    arr = img_to_array(img)
    arr /= 255.0
    if 'ok' in filename:
      label = 1
    else:
      label = 0
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image':
                tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[arr.tobytes()])),
                'label':
                tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

  def write_tfrecord(example):
    filename = os.path.join(FLAGS.dest, create_filename('ok'))
    with tf.python_io.TFRecordWriter(filename) as w:
      w.write(example.SerializeToString())

  try:
    os.makedirs(FLAGS.dest)
  except IOError:
    pass
  files = tf.gfile.Glob(FLAGS.pattern)
  options = PipelineOptions()
  options.view_as(StandardOptions).runner = 'DirectRunner'
  with beam.Pipeline(options=options) as p:
    (p | 'create file names' >> beam.Create(files) |
     'make example' >> beam.Map(mk) |
     'write tfrecord' >> beam.Map(write_tfrecord))


def cropaug():
  try:
    os.makedirs(FLAGS.dest)
  except IOError:
    pass
  options = PipelineOptions()
  options.view_as(StandardOptions).runner = 'DirectRunner'
  with beam.Pipeline(options=options) as p:
    (p | 'read tfrecord by pattern' >> beam.io.ReadFromTFRecord(FLAGS.pattern) |
     'process image' >> RotateParser(features=features, shape=FLAGS.shape) |
     'write tfrecord' >> WriteTFRecord(prefix='ng', dest=FLAGS.dest))


def define_flags():
  flags.DEFINE_string(
      name='pattern', help='TF Rercord file name pattren', default='*.tfrecord')
  flags.DEFINE_string(
      name='dest', help='Destination direcoty of parsed image', default=None)
  flags.DEFINE_list(name='shape', help='Shape of image', default=[224, 224, 3])
  flags.DEFINE_enum(
      name='main',
      help='Main functions',
      enum_values=['main', 'bgrmain', 'image2tfrecord', 'cropaug'],
      default='main')


def run(_):
  m = globals()[FLAGS.main]
  m()


if __name__ == '__main__':
  define_flags()
  FLAGS = flags.FLAGS
  app.run(run)
