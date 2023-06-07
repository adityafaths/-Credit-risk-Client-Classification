import xgboost as xgb
from pathlib import Path
import datetime
import streamlit as st
import pandas as pd
import math

##page layout
st.set_page_config(layout="centered", page_icon="", page_title=" Customer Segmentation In Bank Marketing")
st.title("🏦 Customer Segmentation In Bank Marketing")
st.image("https://echovme.in/wp-content/uploads/2022/07/social-media-digital-marketing-strategies-for-banks.jpg", caption=None)
st.caption("Time deposits are the main source of income for a bank. Time deposits are cash investments held at financial institutions. Your money is invested for an agreed rate of interest over a certain period or period. Telephone marketing campaigns are still one of the most effective ways to reach people. However, they require a large investment of hiring a call center to actually run this campaign. That's why it's so important to identify customers who are most likely to convert first so they can be targeted specifically through calls. The data relates to a direct marketing campaign (phone calls) from a Portuguese banking institution. The purpose of classification is to predict whether a client will subscribe to a time deposit.")
st.subheader(
    ":blue[Will client will subscribe to term deposit?]"
)
st.info("Whilst every effort has been taken during the development of these tools for them to be as accurate and reliable as possible it is important that the user understands they are still a prediction and not an absolute. Any decisions taken whist using these tools are the responsibility of the user and no liability whatsoever will be taken by the developers/authors of the tools or the website owner.",icon="ℹ️")
st.write("##")
st.write("##")

# declaring variables
dict_personal_status= {'male single':0,'female div/dep/mar':1,'male div/sep':2,'male mar/wid':3}
dict_purpose = {'business':0, 'new car':1,'used car':2, 'education':3, 'retraining':4, 'other':5,'domestic appliance':6,'radio/tv':7,'furniture/equipment':8,'repairs':9}
dict_savings_status = {'no known savings':0, '<100':1,'100<=X<500':2,'500<=X<1000':3,'>=1000':4}
dict_own_telephone= {'yes':0, 'none':1}
dict_employment = {'unemployed':0, '<1':1,'1<=X<4':2,'4<=X<7':3,'>=7':4}
dict_property_magnitude = {'no known property':0, 'life insurance':1, 'car':2, 'real estate':3}
dict_checking_status = {'no checking':0, '<0':1, '0<=X<200':2, '>=200':3}
dict_credit_history = {'critical/other existing credit':0, 'delayed previously':1 , 'existing paid':2, 'no credits/all paid':3, 'all paid':4}
dict_other_payment_plans = {'none':0, 'stores':1, 'bank':2}
dict_job = {'unemp/unskilled non res':0, 'unskilled resident':1, 'skilled':2, 'high qualif/self emp/mgmt':3}
dict_housing = {'for free':0, 'rent':1, 'own':2}
dict_other_parties = {'none':0, 'co applicant':1, 'guarantor':2}
dict_installment_commitment = {'1':1, '2':2, '3':3,'4':4}
dict_residence_since = {'1':1, '2':2, '3':3,'4':4}
dict_existing_credits = {'1':1, '2':2, '3':3,'4':4}
# user input

st.subheader("Customer Infomation")

# selecting job
checking_status = st.selectbox(
        "Select checking status",
        (dict_checking_status.keys()),
)

duration = st.number_input(
    "Duration Loan",
    min_value=0
)
credit_history = st.selectbox(
        "Client Credit History ",
        (dict_credit_history.keys()),
)


purpose = st.selectbox(
        "Select Client's purpose",
        (dict_purpose.keys()),
)
credit_amount = st.number_input(
    "credit Amount",
    min_value=0
)

savings_status = st.number_input(
     "Saving status of client",
    dict_savings_status.keys()
)

employment = st.selectbox(
    "Client employed?",
    dict_employment.keys()
)

installment_commitment = st.selectbox(
        "total client installment commitment",
        (dict_installment_commitment.keys()),
)

personal_status = st.selectbox(
        "Client Status",
        (dict_personal_status.keys()),
)


other_parties = st.selectbox(
        "Client have other parties ?",
        (dict_other_parties.keys()),
)

residence_since = st.selectbox(
        "How many years has the client lived in residence? ",
        (dict_residence_since.keys()),
)

property_magnitude = st.selectbox(
        "Client Property Magnitude",
        (dict_property_magnitude.keys()),
)

age = st.number_input(
    "Client Age",
    min_value=0
)

other_payment_plans = st.selectbox(
        "Client Payment plans",
        (dict_other_payment_plans.keys()),
)

housing = st.selectbox(
        "Client Housings",
        (dict_housing.keys()),
)



existing_credits = st.selectbox(
        "Client existing credits",
        (dict_existing_credits.keys()),
)

job = st.selectbox(
        "Client Job",
        (dict_job.keys()),
)
own_telephone = st.selectbox(
        "Client using own telephone?",
        (dict_own_telephone.keys()),
)


# Model
model = xgb.XGBClassifier()
model.load_model('./model.json')

def predict():
    x = [dict_checking_status[checking_status],duration,dict_credit_history[credit_history],dict_purpose[purpose],credit_amount,employment[dict_employment],dict_personal_status[personal_status],dict_other_parties[other_parties],dict_other_parties[other_parties],dict_residence_since[residence_since],dict_property_magnitude[property_magnitude],age,dict_other_payment_plans[other_payment_plans],dict_housing[housing], dict_existing_credits[existing_credits],dict_job[job],dict_own_telephone[own_telephone]]
    df = pd.DataFrame([x] , columns = model.feature_names_in_)
    prediction = model.predict(df)
    prediction_prob = model.predict_proba(df)
    if(prediction[0] == 1):
        st.write('This customer segment :green[will deposit] with probability of ', "%.2f" % (prediction_prob[0][1]*100), '%')
    else:
        st.write('This customer segment :red[will not deposit] with probability of ', "%.2f" % (prediction_prob[0][0]*100), '%')    
        
if st.button('Classify'): 
    predict()
