# https://www.tutorialspoint.com/flask
import numpy as np
#from sklearn.externals import joblib
import joblib
import pandas as pd
import streamlit as st 
#from category_encoders import one_hot
st.title('Drug Marketing and Physician Targeting')

year_quarter=st.selectbox("Select Quarter",
                             options=["201903-Q3", "201904-Q4", "202001-Q1","202002-Q2", "202003-Q3"])

brand_prescribed=st.number_input("brand_prescribed", min_value=0, max_value=1, step=1)
total_representative_visits=st.slider("total_representative_visits", min_value=0, max_value=55, step=1)
total_sample_dropped=st.slider("total_sample_dropped", min_value=0, max_value=1392, step=1)
physician_hospital_affiliation=st.number_input("physician_hospital_affiliation", min_value=0, max_value=1, step=1)
physician_in_group_practice=st.number_input("physician_in_group_practice", min_value=0, max_value=1, step=1)
total_prescriptions_for_indication1=st.slider("total_prescriptions_for_indication1", min_value=0, max_value=2029, step=1)
total_prescriptions_for_indication2=st.slider("total_prescriptions_for_indication2", min_value=0, max_value=2932, step=1)
total_patient_with_commercial_insurance_plan=st.slider("total_patient_with_commercial_insurance_plan", min_value=0, max_value=2109, step=1)
total_patient_with_medicare_insurance_plan=st.slider("total_patient_with_medicare_insurance_plan", min_value=0, max_value=4746, step=1)
total_patient_with_medicaid_insurance_plan=st.slider("total_patient_with_medicaid_insurance_plan", min_value=0, max_value=2538, step=1)
total_competitor_prescription=st.slider("total_competitor_prescription", min_value=0, max_value=8815, step=1)
new_prescriptions=st.slider("new_prescriptions", min_value=0, max_value=3790, step=1)
physician_gender=st.selectbox("Select Gender",
                             options=["M","F"])
physician_speciality=st.selectbox("Select Speciality",
                             options=["nephrology", "urology", "other"])

submit = st.button("Submit")

if submit:
    #st.write("You submitted the form")
    
    data = [{'year_quarter': year_quarter, 'brand_prescribed': brand_prescribed, 'total_representative_visits':total_representative_visits,
             'total_sample_dropped': total_sample_dropped, 'physician_hospital_affiliation': physician_hospital_affiliation, 'physician_in_group_practice':physician_in_group_practice,
             'total_prescriptions_for_indication1': total_prescriptions_for_indication1, 'total_prescriptions_for_indication2': total_prescriptions_for_indication2, 'total_patient_with_commercial_insurance_plan':total_patient_with_commercial_insurance_plan,
             'total_patient_with_medicare_insurance_plan': total_patient_with_medicare_insurance_plan, 'total_patient_with_medicaid_insurance_plan': total_patient_with_medicaid_insurance_plan, 'total_competitor_prescription':total_competitor_prescription,
             'new_prescriptions': new_prescriptions, 'physician_gender': physician_gender, 'physician_speciality':physician_speciality}]
    #st.write(pd.DataFrame(data))   
    xq_point_df=pd.DataFrame(data)
    #print(xq_point_df)
    #st.write("Classifying...")
    #label = predict(uploaded_file)
    #st.write('%s (%.2f%%)' % (label[1], label[2]*100))    
    category_cols= ['physician_gender', 'physician_speciality', 'year_quarter']
    ce_ohe_cat = joblib.load('sklearn_ohe.pkl')
    #xq_point_new = ce_ohe_cat.transform(xq_list_new1)

    xq_list_new2 = ce_ohe_cat.transform(xq_point_df[category_cols])
    xq_list_new2_df = pd.DataFrame(data = xq_list_new2.toarray() )
    
    xq_point_df.reset_index(drop=True, inplace=True)
    xq_list_new2_df.reset_index(drop=True, inplace=True)

    xq_point_final=pd.concat([xq_point_df, xq_list_new2_df], axis=1)
    xq_point_final.drop(['year_quarter','physician_gender','physician_speciality'], axis = 1,inplace = True)
    
    autoscaler = joblib.load('autoscaler.pkl')
    xq_point_new = autoscaler.transform(xq_point_final)
    #rf_model = joblib.load('rf_model.pkl')
    lgbm_model = joblib.load('lgbm_model.pkl')
    #y_pred = rf_model.predict(xq_point_new)
    y_pred = lgbm_model.predict(xq_point_new)
    #y_pred_proba = lgbm_model.predict_proba(xq_point_new)
    #print('Predicted Class for xq point: ',y_pred)
    if (y_pred == [1]):
        y_pred_new='CLASS-1-LOW'
    elif (y_pred == [2]):
        y_pred_new='CLASS-2-MEDIUM'
    elif (y_pred == [3]):
        y_pred_new='CLASS-3-HIGH'
    else: 
        y_pred_new='CLASS-4-VERY_HIGH'
    #print('Predicted Text for xq point: ',y_pred_new)
    st.write("Classifying the point.....")
    st.success(y_pred_new)
