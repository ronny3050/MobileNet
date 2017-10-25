"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
import re
import scipy.io

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def get_images(image_dir):
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  leng = 0
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      tf.logging.warning('No files found')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    file_list = set(file_list)
    file_names = [os.path.basename(file_name) for file_name in file_list]
    leng = leng + len(file_list)
    result[label_name] = {
      'dir': dir_name,
      'testing': file_names,
    }
  return result

def create_model_graph(model_info):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Args:
    model_info: Dictionary containing information about the model architecture.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Graph().as_default() as graph:
    model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              model_info['bottleneck_tensor_name'],
              model_info['resized_input_tensor_name'],
          ]))
  return graph, bottleneck_tensor, resized_input_tensor

def create_model_info(architecture):
  """Given the name of a model architecture, returns information about it.

  There are different base image recognition pretrained models that can be
  retrained using transfer learning, and this function translates from the name
  of a model to the attributes that are needed to download and train with it.

  Args:
    architecture: Name of a model architecture.

  Returns:
    Dictionary of information about the model, or None if the name isn't
    recognized

  Raises:
    ValueError: If architecture name is unknown.
  """
  architecture = architecture.lower()
  if architecture == 'inception_v3':
    # pylint: disable=line-too-long
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    bottleneck_tensor_size = 2048
    input_width = 299
    input_height = 299
    input_depth = 3
    resized_input_tensor_name = 'Mul:0'
    model_file_name = 'classify_image_graph_def.pb'
    input_mean = 128
    input_std = 128
  elif architecture.startswith('mobilenet_'):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
      tf.logging.error("Couldn't understand architecture name '%s'",
                       architecture)
      return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
        version_string != '0.50' and version_string != '0.25'):
      tf.logging.error(
          """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
  but found '%s' for architecture '%s'""",
          version_string, architecture)
      return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
        size_string != '160' and size_string != '128'):
      tf.logging.error(
          """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
          size_string, architecture)
      return None
    if len(parts) == 3:
      is_quantized = False
    else:
      if parts[3] != 'quantized':
        tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
            architecture)
        return None
      is_quantized = True
    data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
    data_url += version_string + '_' + size_string + '_frozen.tgz'
    bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
    bottleneck_tensor_size = 1001
    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    resized_input_tensor_name = 'input:0'
    if is_quantized:
      model_base_name = 'quantized_graph.pb'
    else:
      model_base_name = 'frozen_graph.pb'
    model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
    model_file_name = os.path.join(model_dir_name, model_base_name)
    input_mean = 127.5
    input_std = 127.5
  else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)

  return {
      'data_url': data_url,
      'bottleneck_tensor_name': bottleneck_tensor_name,
      'bottleneck_tensor_size': bottleneck_tensor_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'resized_input_tensor_name': resized_input_tensor_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
  }

def get_model(model):
  with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              'MobilenetV1/Predictions/Reshape:0',
              'input:0',
          ]))
  return bottleneck_tensor, resized_input_tensor

def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]

  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
 
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path

def run_inference_on_image(sess, image_lists, image_data_tensor, image_dir,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    decoded_image_tensor: Output of initial image resizing and preprocessing.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck = []
  
  for label_index, label_name in enumerate(image_lists.keys()):
        for image_index, image_name in enumerate(image_lists[label_name]['testing']):

          image_path = get_image_path(image_lists, label_name, image_index,
                              image_dir, 'testing')
          image_data = gfile.FastGFile(image_path, 'rb').read()
  
          # First decode the JPEG image, resize it, and rescale the pixel values.
          resized_input_values = sess.run(decoded_image_tensor,
                                          {image_data_tensor: image_data})
          # Then run it through the recognition network.
          bottleneck_values = sess.run(bottleneck_tensor,
                                       {resized_input_tensor: resized_input_values})
          bottleneck_values =  np.squeeze(bottleneck_values)
          bottleneck.append(bottleneck_values)
  return bottleneck

def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
  """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer graph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    
  # Gather information about the model archiecture we'll be using.
    #model_info = create_model_info(args.architecture)
    image_list = get_images(args.image_dir)
    class_count = len(image_list.keys())
    if class_count == 0:
      tf.logging.error('No valid folders of images found at ' + args.image_dir)
      return -1
    if class_count == 1:
      tf.logging.error('Only one valid folder of images found at ' +
                     args.image_dir +
                     ' - multiple classes are needed for classification.')
      return -1

    # Load frozen graph
    bottleneck_tensor, input_tensor = get_model(args.model)

    with tf.Session() as sess:
      # Set up the image decoding sub-graph.
      jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(224, 224,
                                                  3, 127.5, 127.5)


      bottleneck_values = run_inference_on_image(sess, image_list, jpeg_data_tensor,
                                                 args.image_dir,
                                                  decoded_image_tensor, input_tensor,
                                                 bottleneck_tensor)
      scipy.io.savemat(args.feat_name,{
              'featsUnnormalized':bottleneck_values,
              'labels':image_list})

          
    
    
##    with tf.Graph().as_default():
##        with tf.Session() as sess:
            # Read the file containing the pairs used for testing
            #pairs = lfw.read_pairs(os.path.expanduser("data/pairs.txt"))

            # Get the paths for the corresponding images
##            paths = lfw.all_paths(args.list, 'png')
##
##            # Load the model
##            print('Model directory: %s' % args.model_dir)
##            meta_file, ckpt_file = helpers.get_model_filenames(os.path.expanduser(args.model_dir))
##
##            print('Metagraph file: %s' % meta_file)
##            print('Checkpoint file: %s' % ckpt_file)
##            helpers.load_model(args.model_dir, meta_file, ckpt_file)
##
##
##            # # Get input and output tensors
##            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
##            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
##            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
##            
##            image_size = images_placeholder.get_shape()[1]
##            embedding_size = embeddings.get_shape()[1]
##            
##            # # Run forward pass to calculate embeddings
##            print('Runnning forward pass on LFW images')
##            batch_size = args.lfw_batch_size
##            nrof_images = len(paths)
##            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
##            emb_array = np.zeros((nrof_images, embedding_size))
##            for i in range(nrof_batches):
##                 start_index = i * batch_size
##                 end_index = min((i + 1) * batch_size, nrof_images)
##                 paths_batch = paths[start_index:end_index]
##                 images = helpers.load_data(paths_batch, False, False, image_size)
##                 feed_dict = {images_placeholder: images, phase_train_placeholder: False}
##                 emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
##
##             
##            sio.savemat(args.feat_name,{'feats':emb_array});


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--architecture',
      type=str,
      default='inception_v3',
      help="""\
      Which model architecture to use. 'inception_v3' is the most accurate, but
      also the slowest. For faster or smaller models, chose a MobileNet with the
      form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
      'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
      pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      for more information on Mobilenet.\
      """)

    parser.add_argument('--image_dir', type=str,
                        help='Path to the data directory containing testing images.')
    parser.add_argument('--model', type=str)
    parser.add_argument('--feat_name', type=str,
                        help='Name of feature mat filename.')
##    parser.add_argument('--list', type=str,
##                        help='The file containing the image paths to use for validation.')
##    parser.add_argument('--lfw_batch_size', type=int,
##                        help='Number of images to process in a batch in the LFW test set.', default=100)
##    parser.add_argument('--model_dir', type=str,
##                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
##    parser.add_argument('--lfw_pairs', type=str,
##                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
##    parser.add_argument('--lfw_file_ext', type=str,
##                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
##    parser.add_argument('--lfw_nrof_folds', type=int,
##                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    
    main(parse_arguments(sys.argv[1:]))
