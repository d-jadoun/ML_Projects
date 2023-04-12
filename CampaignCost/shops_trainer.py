
import tensorflow as tf
# import tensorflow_decision_forests as tfdf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from absl import logging
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from typing import List, Text
import shops_constants
import os

transformed_name = shops_constants.transformed_name
NORMALIZE_TO_0_1 = shops_constants.NORMALIZE_TO_0_1
FLOAT_TO_INT = shops_constants.FLOAT_TO_INT

def model_fn():
    inputs = {}
    for key,item in FLOAT_TO_INT.items():
        inputs[transformed_name(key)] = tf.keras.Input(shape=(1,),name=transformed_name(key))

    for key,item in NORMALIZE_TO_0_1.items():
        inputs[transformed_name(key)] = tf.keras.Input(shape=(1,),name=transformed_name(key))
    # inputs['dental caries_xf'] = tf.keras.Input(shape=(1,),name=transformed_name('dental caries'))
    output = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
    # output = tf.keras.layers.BatchNormalization()(output)
    # output = tf.keras.layers.Dense(4096, activation='relu')(output)
    # for i in range(2):
    output = tf.keras.layers.Dense(30000,activation='relu')(output)
    # output = tf.keras.layers.Dropout(0.05)(output)
    output = tf.keras.layers.BatchNormalization()(output)
    # output = tf.keras.layers.Dense(32, activation='relu')(output)
    output = tf.keras.layers.Dense(1000, activation='relu')(output)
    # output = tf.keras.layers.Dense(2048, activation='relu')(output)
    # output = tf.keras.layers.Dense(1024, activation='relu')(output)
    # output = tf.keras.layers.Dense(2048, activation='relu')(output)
    # output = tf.keras.layers.Dense(2048, activation='relu')(output)
    # output = tf.keras.layers.Dense(512, activation='relu')(output)
    # output = tf.keras.layers.Dense(512, activation='relu')(output)
    # output = tf.keras.layers.Dense(512, activation='relu')(output)
    # output = tf.keras.layers.Dense(1024, activation='relu')(output)
    # output = tf.keras.layers.Dense(1024, activation='relu')(output)
    # output = tf.keras.layers.Dense(1024, activation='relu')(output)
    # output = tf.keras.layers.Dense(1024, activation='relu')(output)
    # output = tf.keras.layers.Dense(1024, activation='relu')(output)
    output = tf.keras.layers.Dense(512, activation='relu')(output)
    # output = tf.keras.layers.Dense(512, activation='relu')(output)
    # output = tf.keras.layers.Dense(512, activation='relu')(output)
    # output = tf.keras.layers.Dense(512, activation='relu')(output)
    # output = tf.keras.layers.Dense(512, activation='relu')(output)
    # output = tf.keras.layers.Dense(512, activation='relu')(output)
#     output = tf.keras.layers.Dense(16, activation='relu')(output)
#     output = tf.keras.layers.Dense(16, activation='relu')(output)
#     output = tf.keras.layers.Dense(16, activation='relu')(output)
#     output = tf.keras.layers.Dense(16, activation='relu')(output)
#     output = tf.keras.layers.Dense(16, activation='relu')(output)
    output = tf.keras.layers.Dense(20,activation='softmax')(output)
    model = tf.keras.models.Model(inputs,output)

    # Compile model
    model.compile(optimizer='adam',#tf.keras.optimizers.Adam(learning_rate=0.1),
                loss='sparse_categorical_crossentropy',#mae
                metrics=['sparse_categorical_accuracy'])

    return model





LABEL_KEY = 'cost_bin_xf'

def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames,compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=2000):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=LABEL_KEY
    )
    return dataset

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Returns a function that parses a serialized tf.Example and applies
    the transformations during inference.
    Args:
        model: The model that we are serving.
        tf_transform_output: The transformation output that we want to 
            include with the model.
    """
    
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")])
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        
        required_feature_spec = {
            k: v for k, v in feature_spec.items() if k != 'cost_bin'
        }

        parsed_features = tf.io.parse_example(
            serialized_tf_examples,
            required_feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)
        output = model(transformed_features)

        return output

    return serve_tf_examples_fn

def run_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(fn_args.train_files,tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files,tf_transform_output)

    model = model_fn()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir),'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,update_freq='batch')

    model.fit(
        x=train_dataset,
        steps_per_epoch = fn_args.train_steps,
        validation_data = eval_dataset,
        validation_steps = fn_args.eval_steps,
        epochs=20,
        callbacks=[tensorboard_callback]
    )
    

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, 
                                 tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples')),
    }
    model.save(fn_args.serving_model_dir,save_format='tf',signatures=signatures,overwrite=True)#
    # print("Model saved to:", fn_args.serving_model_dir)
