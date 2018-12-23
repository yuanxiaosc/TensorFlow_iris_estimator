import numpy as np
import tensorflow as tf


def gen():
    for _ in range(10):
        sz = np.random.randint(3, 20, 1)[0]
        yield np.random.randint(1, 100, sz), np.random.randint(0, 10, 1)[0]


dataset = tf.data.Dataset.from_generator(
    gen, (tf.int32, tf.int32)).repeat(2).shuffle(buffer_size=100).padded_batch(3, padded_shapes=([None], []))
iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()


with tf.Session() as sess:
    try:
        while True:
            _x, _y = sess.run([x, y])
            print("x is :\n", _x)
            print("y is :\n", _y)
            print("*" * 50)

    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        pass