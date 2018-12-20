# TensorFlow_iris_estimator
The main purpose of this resource is to **show how to use tfrecord data** and **customize TensorFlow estimator**. 
Of course, you can modify the code as much as you want. You can also go to [my blog](https://yuanxiaosc.github.io/).

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
