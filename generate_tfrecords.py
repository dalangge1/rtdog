# this script transforms the original data from Kaggle into TFRecord files holding the output of the
# frozen net along with the label, with train and validation separated

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import sys
from random import shuffle

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# model specs
FROZEN_GRAPH = './classify_image_graph_def.pb'
IN_TENSOR = 'DecodeJpeg/contents:0'
OUT_TENSOR = 'pool_3:0' # cut before the fully connected layer

# data location
LABELS_FILE = 'data/labels.csv'
IMAGES_DIR = 'data/train'

VALIDATION_FRAC = 0.15

# output files
TFRECORD_TRAIN_FILE = 'data/train.tfrecord'
TFRECORD_VAL_FILE = 'data/validation.tfrecord'
LABELS_MAP = "data/labels.txt"

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


def get_labels_dataset(slices):
	'''Returns : a dataset of (image_path, class_name)'''
    def parse_row(row):
        v = tf.string_split(tf.expand_dims(row, 0), ',').values
        return v[0], v[1]

    def parse_id(id, c):
        fn = tf.string_join([IMAGES_DIR, '/', id, '.jpg'])
        return fn, c


    dataset = tf.data.Dataset.from_tensor_slices(slices)\
        .skip(1) \
        .map(parse_row) \
        .map(parse_id)
    return dataset

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

for index, slice in enumerate(get_slices()):
    record = TFRECORD_TRAIN_FILE if not index else TFRECORD_VAL_FILE
    print("Generating {}".format(record))

    with tf.Session() as sess, tf.python_io.TFRecordWriter(record) as writer:

        i = 0
        total = slice.eval().shape[0]

        dataset = get_labels_dataset(slice)

        next_it = dataset.make_one_shot_iterator().get_next()


        try:
            while True:
                i += 1
                sys.stdout.write('\r{:0.4f} % : {}'.format(i / total * 100, i))
                sys.stdout.flush()

                fn, label = sess.run(next_it)

                # decode because fn and labels are in binary form
                fn = fn.decode('ascii')
                label = label.decode('ascii')

                label_id = encoder(label)

                # run the image through the net
                img = tf.gfile.GFile(fn, 'rb').read()
                net_output = sess.run(outpt, feed_dict={inpt: img})


                # write (label, output) to tfrecord
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id])),
                    'model_output': tf.train.Feature(bytes_list=tf.train.BytesList(value=[net_output[0].tostring()]))}))

                writer.write(example.SerializeToString())

        except tf.errors.OutOfRangeError:
            print("\nFinished !\n")