
SCALE_TO_Z_SCORE = {'ALT':None,'AST':None,'Cholesterol':None,'Gtp':None,'HDL':None,'LDL':None,'fasting blood sugar':None,'hemoglobin':None,'relaxation':None,'serum creatinine':None,'systolic':None,'triglyceride':None}
NORMALIZE_TO_0_1 = {'age':None,'height(cm)':None,'waist(cm)':None,'weight(kg)':None}
CLIP_AFTER = {'hearing(left)':1,'hearing(right)':1,'eyesight(left)':2,'eyesight(right)':2,'Urine protein':1}
UNCHANGED = {'dental caries':None,'smoking':None}

def transformed_name(key):
    return key + '_xf'
