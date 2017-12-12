import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import sys

NUM_CLASSES = 120
BATCH_SIZE = 10000
EPOCHS = 1000
TFRECORD_FILE = 'data/train.tfrecord'
TFRECORD_VAL_FILE = 'data/validation.tfrecord'
LEARNING_RATE = 0.01

def fc_layer(x, in_dim, out_dim, name=None):
    '''Input : pool_3 from the inception model'''
    with tf.name_scope(name, "FullyConnected", [x]):
        x = tf.reshape(x, [-1, in_dim], name="Flatten")
        w = tf.Variable(tf.truncated_normal([in_dim, out_dim]), name="weights")
        b = tf.Variable(tf.constant(0.1, tf.float32, [out_dim]), name="biases")
        return tf.matmul(x, w) + b



def parse_function(proto):
    features = {
        'label_id': tf.FixedLenFeature([], tf.int64),
        'model_output': tf.FixedLenFeature([], dtype=tf.string)
    }
    parsed_features = tf.parse_single_example(proto, features)

    output = tf.decode_raw(parsed_features['model_output'], tf.float32)
    output = tf.reshape(output, [1, 1, 2048])

    label = tf.cast(parsed_features['label_id'], tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)
    return output, label



# datasets

with tf.name_scope("datasets"):
    train_dataset = tf.data.TFRecordDataset([TFRECORD_FILE])\
        .map(parse_function)\
        .batch(BATCH_SIZE)

    val_dataset = tf.data.TFRecordDataset([TFRECORD_VAL_FILE]) \
        .map(parse_function) \
        .batch(BATCH_SIZE)

    train_iter = train_dataset.make_initializable_iterator()
    val_iter = val_dataset.make_initializable_iterator()


inception_out = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 2048], name="inception_out")
label_one_hot = tf.placeholder(dtype=tf.int32, shape=[None, NUM_CLASSES], name="label_one_hot")

fc_out = fc_layer(inception_out, 2048, 5000)

fc_out = fc_layer(fc_out, 5000, 3000)
fc_out = fc_layer(fc_out, 3000, 3000)
fc_out = fc_layer(fc_out, 3000, 1000)
fc_out = fc_layer(fc_out, 1000, NUM_CLASSES)

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_one_hot, logits=fc_out),
                                   name="CrossEntropy")

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(label_one_hot, 1), predictions=tf.argmax(fc_out, 1), name="acc_metric")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc_metric")
    acc_initializer = tf.variables_initializer(var_list=running_vars)

with tf.name_scope("summaries"):
    tf.summary.scalar('xen', cross_entropy)
    tf.summary.scalar('acc', acc)
    summaries = tf.summary.merge_all()


def compute_val_acc(sess):
    sess.run(val_iter.initializer)
    sess.run(acc_initializer)
    try:
        while True:
            ds_out, ds_label = sess.run(val_iter.get_next())
            sess.run(acc_op, feed_dict={inception_out:ds_out, label_one_hot:ds_label})

    except tf.errors.OutOfRangeError:
        return sess.run(acc)

i = 0
with tf.Session() as sess:
    writer = tf.summary.FileWriter('sum/summaries_fc=5k,3k,3k,120,lr=0.01, bs=10000', sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for e in range(EPOCHS):
        try:
            sess.run(train_iter.initializer)
            while True:
                i += 1

                ds_out, ds_label = sess.run(train_iter.get_next())
                merged, _, xen = sess.run([summaries, train_step, cross_entropy],
                                  feed_dict={inception_out:ds_out, label_one_hot:ds_label})

                writer.add_summary(merged, i)


                val_acc = compute_val_acc(sess)

                sys.stdout.write("\rEpoch: {}, Step: {}, Train loss: {:0.5f}, Accuracy: {:0.5f}".format(e, i, xen, val_acc))
                sys.stdout.flush()

        except tf.errors.OutOfRangeError as e:
            continue

    saver = tf.train.Saver()
    saver.save(sess, "ckpt/model.ckpt")
