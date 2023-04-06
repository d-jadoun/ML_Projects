
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras import layers
import smoking_constants

def model_fn():
    inputs = []
    for key,item in SCALE_TO_Z_SCORE.items():
        inputs.append(tf.keras.Input(shape=(22,),name=transformed_name(key)))

    for key,item in NORMALIZE_TO_0_1.items():
        inputs.append(tf.keras.Input(shape=(22,),name=transformed_name(key)))

    for key,item in CLIP_AFTER.items():
        inputs.append(tf.keras.Input(shape=(22,),name=transformed_name(key)))

    inputs.append(tf.keras.Input(shape-(22,),name=transformed_name('denatal caries')))
    first_out = tf.keras.layers.Dense(512,activation='relu')
    outputs = tf.keras.layers.Dense(1,activation='relu')(first_out)
    # Define model
    model = tf.keras.models.Model(inputs,outputs)

    # Compile model
    model.compile(
    optimizer='adam',
    loss='mae',
    )

    return model





LABEL_KEY = 'smoking_xf'

def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames,compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=32):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=LABEL_KEY
    )
    return dataset

def run_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(fn_args.train_files,tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files,tf_transform_output)

    model = model_fn()

    model.fit(
        train_dataset,
        steps_per_epoch = fn_args.train_steps,
        validation_data = eval_dataset,
        validation_steps = fn_args.eval_steps
    )

    # signatures= {
    #     'serving_default':
    #         _get_serve_tf_examples_fn(
    #             model, tf_transform_output).get_concrete_function(
    #                 tf.TensorSpec(
    #                     shape=[None],
    #                     dtype=tf.string,
    #                     name='examples'
    #                 )
    #             )
    # }
    model.save(fn_args.serving_model_dir,save_format='tf')#,signatures=signatures
