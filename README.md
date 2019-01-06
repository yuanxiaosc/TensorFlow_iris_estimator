## TensorFlow_iris_estimator
The main purpose of this resource is to **show how to use tfrecord data** and **customize TensorFlow estimator**. 

Of course, you can modify the code as much as you want. You can also go to [my blog](https://yuanxiaosc.github.io/).

## tfrecord basic explanation

> For example, see the [code](https://github.com/yuanxiaosc/TensorFlow_iris_tfrecord_and_estimator)

## tensorflow official sample code
[blog_estimators_dataset.py](https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/blog_estimators_dataset.py)
> code explain[Creating Custom Estimators in TensorFlow](https://developers.googleblog.com/2017/12/creating-custom-estimators-in-tensorflow.html)

### Producte tfrecord files

#### 1. Open TFRecordWriter
```
writer = tf.python_io.TFRecordWriter(tfrecord_store_path)
```
#### 2. prepater to feature_dict
```
for index, row in pandas_datafram_data.iterrows():
    dict_row = dict(row)
    feature_dict = {}
    # prepater to feature_dict
    for k,v in dict_row.items():
        if k in ['feature1', 'feature2',...,'featureN']:
            feature_dict[k] = tf.train.Feature(float_list=tf.train.FloatList(value=[v]))
        elif k in ['feature1', 'feature2',...,'featureN']:
            feature_dict[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=[v])
        else:
            feature_dict[k] = tf.train.Feature(int64_list=tf.train.BytesList(value=[v)])
```
#### 3. producte data example
```
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
```
#### 4. serialize to string
```
    serialized = example.SerializeToString()
```
#### 5. write a example to tfrecord file
```
    writer.write(serialized)
```
#### 6. Close TFRecordWriter
```
writer.close()
```

### Using tfrecord files

#### 1. Open TFRecordDataset
```
filenames = ['tfrecord_file_name1','tfrecord_file_name2']
tf_dataset = tf.data.TFRecordDataset(filenames)
```
#### 2. Parse dataset
```
def _parse_function(record):
    features = {"feature1": tf.FixedLenFeature((), tf.float32, default_value=0.0),
                "feature2": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(record, features)
    return {"feature1": parsed_features["feature1"]}, parsed_features["Species"]
dataset = tf_dataset.map(_parse_function)
```
#### 3. Shuffle, repeat, and batch the examples
```
tf_dataset = dataset.shuffle(1000).repeat().batch(batch_size)
```
#### 4. Iterate dataset
```
tf_iterator = tf_dataset.make_one_shot_iterator()
next_element = tf_iterator.get_next()
with tf.Session() as sess:
    for i in range(show_numbers):
        a_data = sess.run(next_element)
        print(a_data)  
```

-----

## About the repo

### Rely on
+ python 3.6
+ TensorFlow 1.12.0

### Usage method
```
python run_iris_estimator.py
```

### Possible output

+ Producte data files
```
store data to data/iris_training.tfrecord
store data to data/iris_test.tfrecord
```
+ Create model
```
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_model_dir': 'model/tf_record_iris', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f2b56401278>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
```
+ Training model
```
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 0 into model/tf_record_iris/model.ckpt.
INFO:tensorflow:loss = 1.9349309, step = 1
INFO:tensorflow:global_step/sec: 84.0476
INFO:tensorflow:loss = 0.13681199, step = 101 (1.195 sec)
INFO:tensorflow:global_step/sec: 73.8782
INFO:tensorflow:loss = 0.06917934, step = 201 (1.349 sec)
INFO:tensorflow:global_step/sec: 80.3087
INFO:tensorflow:loss = 0.07953491, step = 301 (1.246 sec)
INFO:tensorflow:global_step/sec: 82.9652
INFO:tensorflow:loss = 0.08498424, step = 401 (1.205 sec)
INFO:tensorflow:global_step/sec: 82.1202
INFO:tensorflow:loss = 0.06799353, step = 501 (1.218 sec)
INFO:tensorflow:global_step/sec: 90.6455
INFO:tensorflow:loss = 0.064572886, step = 601 (1.103 sec)
INFO:tensorflow:global_step/sec: 80.6642
INFO:tensorflow:loss = 0.046872936, step = 701 (1.239 sec)
INFO:tensorflow:global_step/sec: 88.4239
INFO:tensorflow:loss = 0.041766435, step = 801 (1.131 sec)
INFO:tensorflow:global_step/sec: 82.1831
INFO:tensorflow:loss = 0.055104688, step = 901 (1.217 sec)
```

+ Storage model
```
INFO:tensorflow:Saving checkpoints for 1000 into model/tf_record_iris/model.ckpt.
```

+ Evaluation model
```
INFO:tensorflow:Restoring parameters from model/tf_record_iris/model.ckpt-1000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2018-12-20-12:49:27
INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.96666664, global_step = 1000, loss = 0.06140284
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 1000: model/tf_record_iris/model.ckpt-1000

Test set accuracy: 0.967
```

+ Model prediction
```
INFO:tensorflow:Restoring parameters from model/tf_record_iris/model.ckpt-1000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.

Prediction is "Setosa" (99.8%), expected "Setosa"

Prediction is "Versicolor" (99.9%), expected "Versicolor"

Prediction is "Virginica" (98.1%), expected "Virginica"
```

### Main References
[iris_data.py](https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py)

[custom_estimator.py](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)
