"""Convert image to tfrecord."""

import os

from absl import flags
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='glob', help='Glob pattern of image files', default=None)
flags.DEFINE_integer(name='width', help='Image width', default=224)
flags.DEFINE_integer(name='height', help='Image height', default=224)
flags.DEFINE_string(name='out', help='Output directory', default=None)
flags.DEFINE_enum(
    name='mode',
    help='Output directory',
    default='pillow',
    enum_values=['pillow', 'opencv'])


def make_example(image, label):
  """Make example proto has features of image and label.
  
  Args:
    image: Image bytes.
    label: Label.

  Returns:
    Exmaple proto.
  """
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'image':
              tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
              'label':
              tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
          }))


def write_tfrecord(exmaple, filename):
  """Write exmaple as tfrecord.
  
  Args:
    example: Example proto.
    filename: TFRecord file name.
  """
  with tf.python_io.TFRecordWriter(filename) as w:
    w.write(exmaple.SerializeToString())


def read_file(filename, mode='pillow'):
  """Read image file and convert image to numpy array.
  
  Args:
    filename: Image file name.
    
  Returns:
    Numpy array whose dtype is float32 and value range is [0, 1].
  """
  load_img = None
  if mode == 'pillow':
    load_img = lambda filename: keras.preprocessing.image.load_img(
          filename, target_size=(FLAGS.height, FLAGS.width))
  else:
    load_img = lambda filename: Image.fromarray(cv2.resize(cv2.imread(filename), (FLAGS.height, FLAGS.width)))
  image = load_img(filename)
  arr = keras.preprocessing.image.img_to_array(image)
  return arr / 255.0


def convert(label_fn, file_fn):
  """Convet image files to tfrecords.
  
  Args:
    label_fn: Function creates label from image file name.
    file_fn: Function creates tfrecord file name from image file name.
  """
  files = tf.gfile.Glob(FLAGS.glob)
  [
      write_tfrecord(
          make_example(read_file(f, FLAGS.mode).tobytes(), label_fn(f)),
          file_fn(f)) for f in files
  ]
