import requests
import tensorflow as tf
import json
import base64

# The URL of the prediction endpoint
url = "http://localhost:8501/v1/models/1:predict"


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


example = tf.train.Example(
    features=tf.train.Features(
        feature={
            "avg_cars_at home(approx).1": _float_feature(2.0),
            "coffee_bar": _float_feature(0),
            "florist": _float_feature(0),
            "low_fat": _float_feature(0),
            "num_children_at_home": _float_feature(2.0),
            "prepared_food": _float_feature(0),
            "salad_bar": _float_feature(0),
            "store_sales(in millions)": _float_feature(8.61),
            "store_sqft": _float_feature(36509.0),
            "total_children": _float_feature(2.0),
            "unit_sales(in millions)": _float_feature(3.0),
            "video_store": _float_feature(0),
        }
    )
)
serialized_example = example.SerializeToString()

example_bytes = base64.b64encode(serialized_example).decode("utf-8")
payload = {"signature_name": "serving_default", "instances": [{"b64": example_bytes}]}
response = requests.post(url, json=payload)

# Get the prediction results
prediction_results = json.loads(response.text)
print("Cost:", prediction_results["predictions"])
