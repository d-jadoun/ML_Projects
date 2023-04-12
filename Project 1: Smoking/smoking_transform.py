
import tensorflow as tf
import tensorflow_transform as tft

import smoking_constants

SCALE_TO_Z_SCORE = smoking_constants.SCALE_TO_Z_SCORE
transformed_name = smoking_constants.transformed_name
NORMALIZE_TO_0_1 = smoking_constants.NORMALIZE_TO_0_1
CLIP_AFTER = smoking_constants.CLIP_AFTER
UNCHANGED = smoking_constants.UNCHANGED


def preprocessing_fn(inputs):
    outputs = {}
    for key in SCALE_TO_Z_SCORE.keys():
        key_data = inputs[key]
        scaled_data = tft.scale_to_z_score(key_data)
        clipped_data = tf.clip_by_value(scaled_data, -3.0,3.0)
        outputs[transformed_name(key)] = clipped_data

    for key in NORMALIZE_TO_0_1.keys():
        key_data2 = inputs[key]
        scaled_data2 = tft.scale_to_0_1(key_data2)
        outputs[transformed_name(key)] = scaled_data2

    for key in CLIP_AFTER.keys():
        key_data3 = inputs[key]
        clipped_data3 = tf.clip_by_value(key_data3, 0,CLIP_AFTER[key])
        scaled_data3 = tft.scale_to_0_1(clipped_data3)
        outputs[transformed_name(key)] = scaled_data3

    for key in UNCHANGED.keys():
        key_data4 = inputs[key]
        outputs[transformed_name(key)] = key_data4

    return outputs
