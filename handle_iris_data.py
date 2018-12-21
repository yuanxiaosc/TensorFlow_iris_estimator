import pandas as pd
import numpy as np
import tensorflow as tf
import os
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    """get csv data path"""
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the csv inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset

def store_csv_data():
    """store csv data to waite for conversion to tfrecord data"""
    # It is important to note that csv file cannot contain ordinal columns and headers
    # In other words,pandas_df.to_csv(csv_path, index=False, header=False)
    train_path, test_path = maybe_download()
    if not os.path.exists("data"):
        os.makedirs("data")
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    new_train_path = os.path.join('data',TRAIN_URL.split('/')[-1])
    new_test_path = os.path.join('data',TEST_URL.split('/')[-1])
    train.to_csv(new_train_path, index=False, header=False)
    test.to_csv(new_test_path, index=False, header=False)
    print("store data to {}".format(new_train_path))
    print("store data to {}".format(new_test_path))
    return new_train_path, new_test_path

def store_tfrecord_file():
    """read csv file and conver to tfrecord file"""
    if not os.path.exists("data"):
        os.makedirs("data")
    train_path, test_path = maybe_download()
    csv_train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    csv_test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    def _store_tfrecord(csv_data, file_name):
        store_path = os.path.join('data',file_name)
        print("store data to {}".format(store_path))
        writer = tf.python_io.TFRecordWriter(store_path)
        # read csv file every row, don not contain header
        for index, row in csv_data.iterrows():
            dict_row = dict(row)
            feature_dict = {}
            # prepater to feature_dict
            for k,v in dict_row.items():
                if 'Species' not in k:
                    feature_dict[k] = tf.train.Feature(float_list=tf.train.FloatList(value=[v]))
                else:
                    feature_dict[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=[v.astype(np.int32)]))
            # producte example
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            # serialize to string
            serialized = example.SerializeToString()
            # write a example to tfrecord file
            writer.write(serialized)
        writer.close()
    _store_tfrecord(csv_train,"iris_training.tfrecord")
    _store_tfrecord(csv_test,"iris_test.tfrecord")


def train_input_fn_by_tfrecord(tfrecord_file="data/iris_training.tfrecord", batch_size=100):
    # Convert the tfrecord inputs to a Dataset.
    filenames = [tfrecord_file]
    tf_dataset = tf.data.TFRecordDataset(filenames)
    def _parse_function(record):
        features = {"SepalLength": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                    "SepalWidth": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                    "PetalLength": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                    "PetalWidth": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                    "Species": tf.FixedLenFeature((), tf.int64, default_value=0)}
        parsed_features = tf.parse_single_example(record, features)
        return {"SepalLength": parsed_features["SepalLength"],
                "SepalWidth": parsed_features["SepalWidth"],
                "PetalLength": parsed_features["PetalLength"],
                "PetalWidth": parsed_features["PetalWidth"]}, parsed_features["Species"]
    dataset = tf_dataset.map(_parse_function)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset


def eval_input_fn_by_tfrecord(tfrecord_file="iris_test.tfrecord", batch_size=100):
    """An input function for evaluation"""
    # Convert the tfrecord inputs to a Dataset.
    filenames = [tfrecord_file]
    tf_dataset = tf.data.TFRecordDataset(filenames)
    def _parse_function(record):
        features = {"SepalLength": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                    "SepalWidth": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                    "PetalLength": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                    "PetalWidth": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                    "Species": tf.FixedLenFeature((), tf.int64, default_value=0)}
        parsed_features = tf.parse_single_example(record, features)
        return {"SepalLength": parsed_features["SepalLength"],
                "SepalWidth": parsed_features["SepalWidth"],
                "PetalLength": parsed_features["PetalLength"],
                "PetalWidth": parsed_features["PetalWidth"]}, parsed_features["Species"]
    dataset = tf_dataset.map(_parse_function)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset


def csv_input_fn(csv_path, batch_size,is_traing=True):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    # The remainder of this file contains a simple example of a csv parser,
    #     implemented using the `Dataset` class.
    # `tf.parse_csv` sets the types of the outputs to match the examples given in
    #     the `record_defaults` argument.
    CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
    def _csv_parse_line(line):
        # Decode the line into its fields
        fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
        # Pack the result into a dictionary
        features = dict(zip(CSV_COLUMN_NAMES, fields))
        # Separate the label from the features
        label = features.pop('Species')
        return features, label

    dataset = dataset.map(_csv_parse_line)
    if is_traing:
        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset

def show_memory_dataset(show_numbers=3,batch_size=100):
    """iterate dataset and show data content and type """
    (train_x, train_y), (test_x, test_y) = load_data()
    dataset = train_input_fn(train_x, train_y, batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(show_numbers):
            a_data = sess.run(next_element)
            print(a_data)
            print('^'*50)
    print("array type:\t",a_data[0]['SepalLength'].dtype)
    return a_data

def show_csv_dataset(show_numbers=3,batch_size=100):
    """iterate dataset and show data content and type """
    train_path, test_path = maybe_download()
    csv_dataset = csv_input_fn(train_path, batch_size=batch_size)
    csv_iterator = csv_dataset.make_one_shot_iterator()
    next_element = csv_iterator.get_next()
    with tf.Session() as sess:
        for i in range(show_numbers):
            a_data = sess.run(next_element)
            print(a_data)
            print('^'*50)
    return a_data

def show_tfrecord_dataset(show_numbers=3,batch_size=100):
    """iterate dataset and show data content and type """
    if not os.path.exists("data/iris_training.tfrecord"):
        store_tfrecord_file()
    tf_dataset = train_input_fn_by_tfrecord(
        tfrecord_file="data/iris_training.tfrecord", batch_size=batch_size)
    tf_iterator = tf_dataset.make_one_shot_iterator()
    next_element = tf_iterator.get_next()
    with tf.Session() as sess:
        for i in range(show_numbers):
            a_data = sess.run(next_element)
            print(a_data)
            print('^'*50)
    return a_data

if __name__=="__main__":
    print("show_memory_dataset:\n")
    show_memory_dataset(show_numbers=3,batch_size=5)
    print("show_csv_dataset:\n")
    show_csv_dataset(show_numbers=3,batch_size=5)
    print("show_tfrecord_dataset:\n")
    show_tfrecord_dataset(show_numbers=3,batch_size=5)