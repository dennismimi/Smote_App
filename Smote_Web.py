import streamlit as st  # import the module for the websites
import joblib  # module to load model data
import pandas as pd

st.write("Loan Status Prediction")
education_list = ["Basic", "Primary", "Vocational", "Secondary", "Higher"]
marital_list = ["Married", "Cohabitant", "Single", "Divorced", "Widow"]
employmentstatus = ["Unemployed", "Partially", "Fully", "Self", "Entrepreneur", "Retiree"]
hometypeownership = ["Homeless", "Owner", "Living_With_Parents", "Tenant", "Prefurnished_Property",
                     "Unfurnished_Property", "Joint_Tenant", "Joint_Ownership", "Mortgage", "Owner_with_Encumbrance",
                     "Other"]
occupationarea = ["Other", "Mining", "Processing", "Energies", "Utilities", "Construction",
                  "Retail_and_Wholesale", "Transport_and_Warehousing", "Hospitality_and_Catering",
                  "Finance_and_Insurance", "Real_Estate", "Research", "Administrative", "Civil_Service_and_Military",
                  "Education", "Health_Care_and_Social_Help", "Arts_and_Entertainment",
                  "Agriculture_Forestry_and_Fishing"]
useofloan = ["Not_Set", "Loan_Consolidation", "Real_Estate", "Home_Improvement", "Business", "Education", "Travel",
             "Vehicle", "Other", "Health", "Finance_and_Insurance", "Research", "Administrative",
             "Civil_Service_and_Military", "Education_2", "Health_Care_and_Social_Help", "Arts_and_Entertainment",
             "Agriculture_Forestry_and_Fishing"]
loanduration = ["less_than_a_month", "1 Month", "2_Months", "3_Months", "4_Months", "5_Months", "6_Months"]
col1, col2, col3 = st.columns(3)
Gender = col1.selectbox("Enter your gender", ["Male", "Female"])
Age = col2.number_input("Enter your age")
MaritalStatus = col1.selectbox("Enter your gender", marital_list)
Education = col3.selectbox("Highest academic qualification", education_list)
NewCreditCustomer = col1.selectbox("Is this your first time Applying for a loan?", ["Yes", "No"])
EmploymentStatus = col3.selectbox("what job do you do?", employmentstatus)
UseOfLoan = col3.selectbox("What is the use of the loan?", useofloan)
HomeOwnershipType = col1.selectbox("what is you status of your home?", hometypeownership)
OccupationArea = col1.selectbox("What area do you stay?", occupationarea)
# EmploymentDurationCurrentEmployer = col2.number_input("How long have you been employed? ")
LoanDuration = col2.selectbox("how long will you take to pay the loan", loanduration)
AppliedAmount = col2.number_input("The amount you wish to apply")
Interest = col3.number_input("The percentage of interest")
Amount = col2.number_input("The amount you got")
Rating = 3.86
ExistingLiabilities = 2.57
DebtToIncome = 4.16
IncomeTotal = 2036
Restructured = 0.32
EmploymentDurationCurrentEmployer = 3.5
NoOfPreviousLoansBeforeLoan = 1.5
CreditScoreEsMicroL = 1.6
ModelVersion = 5.26
VerificationType = 3.35
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

df_pred['Gender'] = df_pred['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

df_pred['NewCreditCustomer'] = df_pred['NewCreditCustomer'].apply(lambda x: 1 if x == 'Yes' else 0)


def transform(data):
    if data in marital_list:
        return marital_list.index(data)
    elif data in employmentstatus:
        return employmentstatus.index(data)
    elif data in education_list:
        return education_list.index(data)
    elif data in occupationarea:
        return occupationarea.index(data)
    elif data in useofloan:
        return useofloan.index(data)
    else:
        return loanduration.index(data)


df_pred['Education'] = df_pred['Education'].apply(transform)
df_pred['EmploymentStatus'] = df_pred['EmploymentStatus'].apply(transform)
df_pred['MaritalStatus'] = df_pred['MaritalStatus'].apply(transform)
df_pred['HomeOwnershipType'] = df_pred['HomeOwnershipType'].apply(transform)
df_pred['UseOfLoan'] = df_pred['UseOfLoan'].apply(transform)
df_pred['LoanDuration'] = df_pred['LoanDuration'].apply(transform)
df_pred['OccupationArea'] = df_pred['OccupationArea'].apply(transform)

model = joblib.load('catboost_model.pkl')
prediction = model.predict(df_pred)
st.write(prediction)

if st.button('Predict'):
    if prediction[0] == 0:
        st.write('<p class="big-font">You are a defaulter.</p>',
                 unsafe_allow_html=True)
    else:
        st.write('<p class="big-font">You have paid you loan.</p>', unsafe_allow_html=True)
