import tensorflow as tf
import os
import generate_tfrecords
from PIL import Image
import numpy as np

inpt, inception_out = generate_tfrecords.net()

def fc_layer(x, in_dim, out_dim, name=None):
    '''Input : pool_3 from the inception model'''
    with tf.name_scope(name, "FullyConnected", [x]):
        x = tf.reshape(x, [-1, in_dim], name="Flatten")
        w = tf.Variable(tf.truncated_normal([in_dim, out_dim]), name="weights")
        b = tf.Variable(tf.constant(0.1, tf.float32, [out_dim]), name="biases")
        return tf.matmul(x, w) + b

fc_out = fc_layer(inception_out, 2048, 5000)

fc_out = fc_layer(fc_out, 5000, 5000)
fc_out = fc_layer(fc_out, 5000, 5000)
fc_out = fc_layer(fc_out, 5000, 120)

out_soft = tf.nn.softmax(fc_out)

with tf.Session() as sess, open('predict.csv', 'w') as csv:
    saver = tf.train.Saver()
    saver.restore(sess, "ckpt/model.ckpt")

    for file in os.listdir("data/test"):
        fn = "data/test/"+file
        id = file.split(".")[0]
        img = Image.open(fn)

        out = sess.run(out_soft, feed_dict={inpt: img})
        print(id, np.argmax(out))

        csv.write(id)
        for a in out[0]:
            csv.write(', '+str(a))

        csv.write("\n")
