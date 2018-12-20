"""An Example of a custom Estimator using tfrecord for the Iris dataset."""
import argparse
import tensorflow as tf
from TensorFlow_iris_estimator import handle_iris_data
from TensorFlow_iris_estimator.iris_estimator_model import my_model_example_one as my_model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--model_dir', default="model/tfrecord_iris", type=str, help="store model location")

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data convert to tfrecord format and store
    handle_iris_data.store_tfrecord_file()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in CSV_COLUMN_NAMES[:-1]:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        },
        model_dir=args.model_dir
    )

    # Train the Model.
    classifier.train(
        input_fn=lambda: handle_iris_data.train_input_fn_by_tfrecord(
            tfrecord_file="data/iris_training.tfrecord",batch_size=args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: handle_iris_data.eval_input_fn_by_tfrecord(
            tfrecord_file="data/iris_test.tfrecord", batch_size=args.batch_size),
    )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:handle_iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(handle_iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
