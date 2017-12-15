# this script transforms the original data from Kaggle into TFRecord files holding the output of the
# frozen net along with the label, with train and validation separated

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import sys
from random import shuffle
from enum import Enum
from PIL import Image
import numpy as np
from io import BytesIO

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# model specs
FROZEN_GRAPH = './classify_image_graph_def.pb'
IN_TENSOR = 'Cast:0' # we bypass the jpeg decode because PIL does it for us
# IN_TENSOR = 'DecodeJpeg/contents:0' # you can pass directly the jpeg bytes if you don't do any transformation
OUT_TENSOR = 'pool_3:0' # cut before the fully connected layer

# data location
LABELS_FILE = 'data/labels.csv'
IMAGES_DIR = 'data/train'

VALIDATION_FRAC = 0.15

# output files
TFRECORD_TRAIN_FILE = 'data/train.tfrecord'
TFRECORD_VAL_FILE = 'data/validation.tfrecord'
LABELS_MAP = "data/labels.txt"

ROTATION_AMOUNT = 25 # deg, for data augmentation

class Transf(Enum):
    NONE = 1
    CROP = 2 # crop a center square
    FLIP = 3 # flip the image horizontally
    ROT_L = 4 # rotate left by ROTATION_AMOUNT
    ROT_R = 5 # rotate right by ROTATION_AMOUNT


def get_color_mean_std_dev():
    '''Runs over all the data and returns the mean and std dev for each color R, G, B'''
    mean = np.zeros(3)
    std_dev = np.zeros(3)
    i = 0
    for fn in os.listdir(IMAGES_DIR):
        i += 1
        img = np.array(Image.open(os.path.join(IMAGES_DIR, fn)))
        mean += np.mean(img, (0, 1))
        std_dev += np.mean(np.square(img), (0, 1))

    mean /= i
    std_dev /= i - 1
    std_dev -= np.square(mean)
    std_dev = np.sqrt(std_dev)

    return mean, std_dev


def net():
    '''Returns : input tensor, output tensor, cut accordingly to constants above'''
    with tf.gfile.GFile(FROZEN_GRAPH, 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())
    inpt, outpt = tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=[IN_TENSOR, OUT_TENSOR],
        name='InceptionV3'
        )
    return inpt, outpt


def get_labels_datasets():
    '''Returns : a dataset of (image_path, class_name, transf)'''
    def parse_row(row):
        v = tf.string_split(tf.expand_dims(row, 0), ',').values
        return v[0], v[1]

    def parse_id(id, c):
        fn = tf.string_join([IMAGES_DIR, '/', id, '.jpg'])
        return fn, c

    def parse_transf_val(id, c):
        return id, c, Transf.NONE.value

    def parse_transf_train(id, c):
        dataset = None
        for t in Transf:
            d = tf.data.Dataset.from_tensors((id, c, t.value))
            if not dataset:
                dataset = d
            else:
                dataset = dataset.concatenate(d)

        return dataset


    train_s, val_s = get_slices()


    dataset_train = tf.data.Dataset.from_tensor_slices(train_s)\
                    .skip(1) \
                    .map(parse_row) \
                    .map(parse_id) \
                    .flat_map(parse_transf_train)

    dataset_val = tf.data.Dataset.from_tensor_slices(val_s) \
                    .skip(1) \
                    .map(parse_row) \
                    .map(parse_id) \
                    .map(parse_transf_val)

    return dataset_train, dataset_val

def label_encoder():
    '''Encode class name to a integer, and writes the mapping to a file.
	Returns a function to encode any class name'''
    df = pd.read_csv(LABELS_FILE).replace(' ', '_')
    df = df.iloc[:, 1].drop_duplicates().map(lambda x: x.rstrip())
    df.index = range(len(df))
    df.to_csv(LABELS_MAP, header=False, index=True, sep=':')

    def encode(label):
        label = label.rstrip().replace(' ', '_')
        return df[df == label].index.tolist()[0]

    return encode

def get_slices():
    '''Slices the lines of the labels file into 2 slices : validation and train'''
    with open(LABELS_FILE, 'r') as fp:
        lines = fp.readlines()
    lines.pop(0)
    shuffle(lines)
    assert 0 <= VALIDATION_FRAC <= 1
    num_train = int(len(lines) * (1 - VALIDATION_FRAC))
    return [tf.constant(lines[:num_train], dtype=tf.string), tf.constant(lines[num_train+1:], dtype=tf.string)]

def transform(img, transform):
    if transform == Transf.NONE:
        return img
    elif transform == Transf.CROP:
        m = min(img_pil.size[0], img_pil.size[1])
        return img.crop(((img.size[0] - m) / 2, (img.size[1] - m) / 2,
                         (img.size[0] + m) / 2, (img.size[1] + m) / 2))
    elif transform == Transf.FLIP:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == Transf.ROT_L:
        return img.rotate(ROTATION_AMOUNT, Image.BILINEAR, expand=False)
    elif transform == Transf.ROT_R:
        return img.rotate(-ROTATION_AMOUNT, Image.BILINEAR, expand=False)
    else:
        return img


print(
"""
Generating TFRecords...

Parameters:
 -- Frozen graph path: {}
 -- Input tensor: {}
 -- Output tensor: {}
 -- Labels file: {}
 -- Images directory: {}
 -- Validation size: {}

Will be generated:
 -- Train TFRecord: {}
 -- Validation TFRecord: {}
 -- Labels map file: {}
""".format(FROZEN_GRAPH, IN_TENSOR, OUT_TENSOR, LABELS_FILE, IMAGES_DIR,
           VALIDATION_FRAC, TFRECORD_TRAIN_FILE, TFRECORD_VAL_FILE, LABELS_MAP))


encoder = label_encoder()
inpt, outpt = net()

with open(LABELS_FILE, 'r') as fp:
    lines = fp.readlines()

total_lines = len(lines)

for index, dataset in enumerate(get_labels_datasets()):
    record = TFRECORD_TRAIN_FILE if not index else TFRECORD_VAL_FILE
    print("Generating {}".format(record))

    with tf.Session() as sess, tf.python_io.TFRecordWriter(record) as writer:

        i = 0
        if not index:
            total = total_lines * (1- VALIDATION_FRAC) * len([t for t in Transf])
        else:
            total = total_lines * VALIDATION_FRAC

        next_it = dataset.make_one_shot_iterator().get_next()


        try:
            while True:
                i += 1
                sys.stdout.write('\r{:0.4f} % : {}'.format(i / total * 100, i))
                sys.stdout.flush()

                fn, label, t = sess.run(next_it)

                # decode because fn and labels are in binary form
                fn = fn.decode('ascii')
                label = label.decode('ascii')


                img_pil = Image.open(fn)

                img = transform(img_pil, Transf(t))



                label_id = encoder(label)

                # run the image through the net
                #img = tf.gfile.GFile(fn, 'rb').read()

                net_output = sess.run(outpt, feed_dict={inpt: img})


                # write (label, output) to tfrecord
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id])),
                    'model_output': tf.train.Feature(bytes_list=tf.train.BytesList(value=[net_output[0].tostring()]))}))

                writer.write(example.SerializeToString())

        except tf.errors.OutOfRangeError:
            print("\nFinished !\n")