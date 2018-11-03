"""Run."""

import os

from absl import app
from absl import flags

import img_to_tfrecord

FLAGS = flags.FLAGS


def label_fn(filename):
  label = 1
  if '/ng/' in filename:
    label = 0
  return label


def file_fn(filename):
  cat = 'ng' if '/ng/' in filename else 'ok'
  tfrecord_name = '.'.join(os.path.basename(filename).split('.')[:-1])
  return os.path.join(FLAGS.out, cat, '{}-{}.tfrecord'.format(
      cat, tfrecord_name))


def main(_):
  try:
    os.makedirs(os.path.join(FLAGS.out, 'ok'))
    os.makedirs(os.path.join(FLAGS.out, 'ng'))
  except IOError:
    pass
  img_to_tfrecord.convert(label_fn, file_fn)


if __name__ == '__main__':
  app.run(main)
