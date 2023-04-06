
import tensorflow as tf
import tensorflow_transform as tft

import shops_constants

transformed_name = shops_constants.transformed_name
NORMALIZE_TO_0_1 = shops_constants.NORMALIZE_TO_0_1
FLOAT_TO_INT = shops_constants.FLOAT_TO_INT
LABEL_CHANGE = shops_constants.LABEL_CHANGE

def preprocessing_fn(inputs):
    outputs = {}

    for key in FLOAT_TO_INT.keys():
        key_data3 = inputs[key]
        changed_data3 = tf.cast(key_data3,dtype=tf.int64)
        outputs[transformed_name(key)] = changed_data3

    for key in NORMALIZE_TO_0_1.keys():
        key_data2 = inputs[key]
        scaled_data2 = tft.scale_to_0_1(key_data2)
        outputs[transformed_name(key)] = scaled_data2
    
    for key in LABEL_CHANGE.keys():
        key_data4 = inputs[key]
        scaled_data4 = key_data4#tft.scale_to_0_1(key_data4)
        outputs[transformed_name(key)] = scaled_data4


    return outputs
