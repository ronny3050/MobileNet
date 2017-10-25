# MobileNet ![You_asked for_this](https://img.shields.io/badge/You_asked-for_this-orange.svg)

> MobileNet with ability to fine-tune and incorporate center or triplet loss

A tensorflow implementation of Google's [MobileNets](https://arxiv.org/abs/1704.04861) for re-training/fine-tuning on your own custom dataset with the addition of (optional) center loss or triplet loss. Additionally, this repo can be used to re-train Inception network as well with the above added benefits.

## Tensorflow release
Currently this repo is compatible with Tensorflow 1.3.0.

## News
| Date     | Update |
|----------|--------|
| 2017-10-25 | Currently working on triplet loss |
| 2017-10-25 | Added code to support center loss |

## Pre-trained Model
Inception_v3 is the most accurate model, but also the slowest. For faster or smaller models, chose a MobileNet with the form `mobilenet_<parameter_size>_<input_size>_[(optional)quantized]`. For example,'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
     pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images.

## Inspiration
The code is heavily inspired by the Tensorflow's [Retrain Script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py) and [FaceNet](https://github.com/davidsandberg/facenet).




