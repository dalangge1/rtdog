import tensorflow as tf

TFRECORD_FILE = 'data/train.tfrecord'
BATCH_SIZE = 5000

def parse_function(proto):
    features = {
        'label_id': tf.FixedLenFeature([], tf.int64),
        'model_output': tf.FixedLenFeature([], dtype=tf.string)
    }
    parsed_features = tf.parse_single_example(proto, features)

    output = tf.decode_raw(parsed_features['model_output'], tf.float32)
    label = tf.cast(parsed_features['label_id'], tf.int32)


    output = tf.reshape(output, [1, 1, 2048])
    return output, label

dataset = tf.data.TFRecordDataset([TFRECORD_FILE])\
            .map(parse_function)\
            .batch(BATCH_SIZE)


iter = dataset.make_one_shot_iterator()
next = iter.get_next()
i = 0
with tf.Session() as sess:
    try:
        while True:
            o, l = sess.run(next)
            i += 1
            print(i)
    except tf.errors.OutOfRangeError:
        print("end")