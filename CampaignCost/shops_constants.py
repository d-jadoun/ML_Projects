
NORMALIZE_TO_0_1 = {'store_sales(in millions)':None,'store_sqft':None}
FLOAT_TO_INT = {'avg_cars_at home(approx).1':None,'coffee_bar':None,'florist':None,'salad_bar':None,'unit_sales(in millions)':None,'video_store':None,'total_children':None,'num_children_at_home':None,'low_fat':None,'prepared_food':None}
LABEL_CHANGE = {'cost_bin':None}

def transformed_name(key):
    return key + '_xf'
