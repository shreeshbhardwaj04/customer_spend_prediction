#streamlit UI
import pickle
import pandas as pd
import streamlit as st

#page config
st.set_page_config(page_title="Customer spending prediction",
                   page_icon="logo.png",
                   layout="wide")


#sidebar
with st.sidebar:
    st.title("Customer Spending Prediction")
    st.image("logo.png")

#load the model
with open("best_model_pipeline.pkl","rb") as file:
    best_pipeline = pickle.load(file)

#load the cleaned dataset
df = pd.read_csv("cleaned_data.csv")

#user input

with st.container(border=True):
    col1,col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age: ",min_value=1,
                            max_value=100)
        Annual_Income = st.number_input("Annual income: ",
                                        min_value=150000)
        Website_Visits = st.number_input("Website visits: ",
                                        min_value=1)

        Previous_Purchases = st.number_input("Previous Purchases: ",
                                            min_value=0)


    with col2:
        Membership_Type = st.selectbox("Membership Type: ",
                                    options=df["Membership_Type"].unique())
        Preferred_Category = st.selectbox("Preferred category: ",
                                    options=df["Preferred_Category"].unique())
        City_Tier = st.selectbox("City Tier: ",
                                options=df["City_Tier"].unique())
        Discount_Used =st.selectbox("Discount Used: ",
                                    options=df["Discount_Used"].unique())


#new data
new_data = {
    'Age':Age, 'Annual_Income':Annual_Income, 'Website_Visits':Website_Visits, 'Previous_Purchases':Previous_Purchases,
       'Membership_Type':Membership_Type, 'Preferred_Category':Preferred_Category, 'City_Tier':City_Tier,
         'Discount_Used':Discount_Used
}

new_data = pd.DataFrame(new_data,index=[0])

if st.button("Predict Spend"):
    prediciton = best_pipeline.predict(new_data)[0]
    prediciton = round(prediciton,ndigits=2)
    st.header(f"Predicted Spend: {prediciton} Rs.")