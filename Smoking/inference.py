import requests
import tensorflow as tf
import json
import base64
# The URL of the prediction endpoint
url = 'http://localhost:8501/v1/models/6:predict'

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

example = tf.train.Example(features=tf.train.Features(feature={
'ALT': _int64_feature(0),
'AST': _int64_feature(0),
'Cholesterol': _int64_feature(0),
'Gtp': _int64_feature(0),
'HDL': _int64_feature(0),
'LDL': _int64_feature(0),
'fasting blood sugar': _int64_feature(0),
'hemoglobin': _float_feature(0),
'relaxation': _int64_feature(0),
'serum creatinine': _float_feature(0),
'systolic': _int64_feature(0),
'triglyceride': _int64_feature(0),
'age': _int64_feature(0),
'height(cm)': _int64_feature(0),
'waist(cm)': _float_feature(0),
'weight(kg)': _int64_feature(0),
'hearing(left)': _int64_feature(0),
'hearing(right)': _int64_feature(0),
'eyesight(left)': _float_feature(0),
'eyesight(right)': _float_feature(0),
'Urine protein': _int64_feature(0),
'dental caries': _int64_feature(0)
}))
serialized_example = example.SerializeToString()

example_bytes = base64.b64encode(serialized_example).decode('utf-8')
payload = {
    'signature_name': 'serving_default',
    'instances': [{
        'b64': example_bytes
    }]
}
response = requests.post(url, json=payload)

# Get the prediction results
prediction_results = json.loads(response.text)
print('Smoking Status:', bool(round(prediction_results['predictions'][0][0])))
