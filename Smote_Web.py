import streamlit as st  # import the module for the websites
import joblib  # module to load model data
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import pickle



#------- SETTINGS-----------
page_title = "Credit Scoring Application"
page_write = " Fill the Form Below for Prediction"
page_icon = ":moneybag:"
layout = "centered"
#-----------------




st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

st.title(page_title + " " + page_icon)

st.write(""" ### Fill the Form Below for Prediction:money_with_wings:""")

st.sidebar.selectbox('Explore or Predict',("Predict","Explore"))

Education_list = ["Basic", "Primary", "Vocational", "Secondary", "Higher"]
Marital_list = ["Married", "Cohabitant", "Single", "Divorced", "Widow"]
EmploymentStatus = ["Unemployed", "Partially", "Fully", "Self", "Entrepreneur", "Retiree"]
HomeTypeOwnership = ["Homeless", "Owner", "Living_With_Parents", "Tenant", "Prefurnished_Property",
                     "Unfurnished_Property", "Joint_Tenant", "Joint_Ownership", "Mortgage", "Owner_with_Encumbrance",
                     "Other"]
OccupationArea = ["Other", "Mining", "Processing", "Energies", "Utilities", "Construction",
                  "Retail_and_Wholesale", "Transport_and_Warehousing", "Hospitality_and_Catering",
                  "Finance_and_Insurance", "Real_Estate", "Research", "Administrative", "Civil_Service_and_Military",
                  "Education", "Health_Care_and_Social_Help", "Arts_and_Entertainment",
                  "Agriculture_Forestry_and_Fishing"]
UseOfLoan = ["Not_Set", "Loan_Consolidation", "Real_Estate", "Home_Improvement", "Business", "Education", "Travel",
             "Vehicle", "Other", "Health", "Finance_and_Insurance", "Research", "Administrative",
             "Civil_Service_and_Military", "Education_2", "Health_Care_and_Social_Help", "Arts_and_Entertainment",
             "Agriculture_Forestry_and_Fishing"]
LoanDuration = ["less_than_a_month", "1 Month", "2_Months", "3_Months", "4_Months", "5_Months", "6_Months"]

col1, col2, = st.columns(2)

Age = col1.slider('Enter Age',0,100)
Gender = col2.radio("Select Gender",["Male", "Female", "Other"])
MaritalStatus = col1.selectbox("Choose which best describes your Marital Status",Marital_list)
Education = col1.selectbox("Education Level", Education_list)
NewCreditCustomer = col2.select_slider('Is This Your First Time Applying for Credit', ['Yes', 'No'])
EmploymentStatus = col1.selectbox('Choose Which best describes your Employment Status', EmploymentStatus)
HomeOwnershipType = col2.selectbox('Choose Which Best Describes Your Home',HomeTypeOwnership)
EmploymentDurationCurrentEmployer = col1.slider('How Long have you been Employed',0,50)
OccupationArea = col2.selectbox('Select Your Occupation Area', OccupationArea)
LoanDuration = col1.select_slider("How long do you intend to take to pay off your credit", LoanDuration)
Interest = col2.number_input('The percentage of Interest')
IncomeTotal = col1.number_input('Whats your Total Income?')
NoOfPreviousLoansBeforeLoan = col2.slider('Number of Previous Loans',1,20)
AppliedAmount = col2.number_input("The amount you wish to apply", 12345)
Amount = st.number_input('Amount you Received',10000)

Rating = 10
ExistingLiabilities = 0
DebtToIncome = 0.12
Restructured = 1
CreditScoreEsMicroL = 0.9
ModelVersion = 2
VerificationType = 1
LanguageCode = 9600
df_pred = pd.DataFrame([[Age, LoanDuration, NewCreditCustomer, VerificationType, Gender,
                         AppliedAmount, Interest, UseOfLoan, Amount, Education,
                         EmploymentDurationCurrentEmployer, Rating, MaritalStatus,
                         EmploymentStatus, OccupationArea, HomeOwnershipType, ExistingLiabilities,
                         DebtToIncome, IncomeTotal,
                         Restructured, NoOfPreviousLoansBeforeLoan, CreditScoreEsMicroL, ModelVersion]],

                       columns=['Age', 'LoanDuration', 'NewCreditCustomer', 'VerificationType', 'Gender',
                                'AppliedAmount', 'Interest', 'UseOfLoan', 'Amount', 'Education',
                                'EmploymentDurationCurrentEmployer', 'Rating', 'MaritalStatus',
                                'EmploymentStatus', 'OccupationArea', 'HomeOwnershipType', 'ExistingLiabilities',
                                'DebtToIncome', 'IncomeTotal',
                                'Restructured', 'NoOfPreviousLoansBeforeLoan', 'CreditScoreEsMicroL', 'ModelVersion'])
def transform(data):
    if data in Marital_list:
        return Marital_list.index(data)
    elif data in Education_list:
        return Education_list.index(data)
    elif data in EmploymentStatus:
        return EmploymentStatus.index(data)
    elif data in OccupationArea:
        return OccupationArea.index(data)
    elif data in UseOfLoan:
        return UseOfLoan.index(data)
    elif data in LoanDuration:
        return LoanDuration.index(data)
    else:
        return 0
df_pred['Gender'] = df_pred['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df_pred['NewCreditCustomer'] = df_pred['NewCreditCustomer'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['Education'] = df_pred['Education'].apply(transform)
df_pred['EmploymentStatus'] = df_pred['EmploymentStatus'].apply(transform)
df_pred['MaritalStatus'] = df_pred['MaritalStatus'].apply(transform)
df_pred['HomeOwnershipType'] = df_pred['HomeOwnershipType'].apply(transform)
df_pred['UseOfLoan'] = df_pred['UseOfLoan'].apply(transform)
df_pred['LoanDuration'] = df_pred['LoanDuration'].apply(transform)
df_pred['OccupationArea'] = df_pred['OccupationArea'].apply(transform)

def make_predictions (pred_data):
    
    model = joblib.load('catboost_model.pkl')
    pred = model.predict(pred_data)
    return pred
                      
    
prediction = make_predictions (df_pred)
st.write(prediction) 
if st.button('Predict'):
    if prediction[0] == 0:
        st.write('<p class="big-font">Your Credit Score is High.You can apply for a Loan.:thumbsup:</p>',
                unsafe_allow_html=True)
    else:
        st.write('<p class="big-font">Your Credit Score is Low.You are Likely to Default.:thumbsdown:</p>',
                unsafe_allow_html=True)
