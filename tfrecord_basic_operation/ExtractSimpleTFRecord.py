# Demonstration of extracting TFRecord file with simple data types like int, float, string

import tensorflow as tf

tf.reset_default_graph()

def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        'int_list1': tf.FixedLenFeature([], tf.int64),
        'float_list1': tf.FixedLenFeature([], tf.float32),
        'str_list1': tf.FixedLenFeature([], tf.string),
        # If size is different of different records, use VarLenFeature 
        'float_list2': tf.VarLenFeature(tf.float32),
        'image_list': tf.FixedLenFeature([], tf.string)
    }
    sample = tf.parse_single_example(data_record, features)
    sample['float_list2'] = tf.sparse.to_dense(sample['float_list2'])
    sample['image_list'] = tf.decode_raw(sample['image_list'], tf.float64)
    sample['image_list'] = tf.reshape(sample['image_list'], (3, 4, 2))
    return sample

# Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(['example.tfrecord'])
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    try:
        while True:
            data_record = sess.run(next_element)
            print(data_record)
    except:
        pass

