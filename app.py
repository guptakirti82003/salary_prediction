import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,r2_score,precision_score
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Salary_predictor",page_icon="",layout="wide")

########################### lottie files ###########################################

def load_lottiefile(filepath: str): #Applying animation from json file
    with open(filepath,"r",encoding='cp850') as f:
        return json.load(f)

money_animate=load_lottiefile("lottiefiles/money.json")
strategy_animate=load_lottiefile("lottiefiles/strategy.json")
technology_animate=load_lottiefile("lottiefiles/technology.json")
welcome_animate=load_lottiefile("lottiefiles/welcome.json")
greet_animate=load_lottiefile("lottiefiles/greet.json")

df1=pd.read_csv("salary.csv")
df2=pd.read_csv("top_companies_salary.csv")

############################# Starting of an programs ###################################
st.markdown("<h1 style='text-align:center; color:white;'>SALARY PREDICTOR </h1>",True)
st.write("---")
nav=option_menu(
    menu_title="",
    options=["ABOUT PROJECT","DATA ANALYZE","PREDICT SALARY"],
    menu_icon="cast",
    icons=["easel2-fill","file-earmark-ppt-fill","info-circle","file-person"],
    orientation="horizontal",
    default_index=0,
    styles={
            # "container": {"max-width": "100000px", "background-color": "#fafafa"},
            # "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"--hover-color": "#008b8b"},
            # "nav-link-selected": {"background-color": "#02ab21"},
        },
    )
st.write("---")

############################### OPTION 1 ########################################
if nav=="ABOUT PROJECT":
    st.markdown("<h1 style='text-align:center; color:white;'>OVERVIEW</h1>",True)
    st.markdown("<h4 style='text-align:center; color:white;'>This is project predicts the salary of the employee based on the experience, years at company, gender , Type of Company and Profession.</h4>",True)
    left_col,right_col=st.columns((4,5))
    with left_col:
        st_lottie(money_animate,speed=1,loop=True,quality="low")
    with right_col:
        st.markdown("<h1 style='text-align:left; color:white;'><u>PROJECT INFO</u></h1>",True)
        st.write(
                """
                This project is based on Machine Learning.\n
                In this project,it will simply predict the salary on the basis of 
                given data.\n
                The purpose of this project is to use data transformation and machine learning to create a model
                that will predict a salary when given years of experience, years at company, gender ,selection of company and profession.
                """
                )
    st.write("---")
    left_col,right_col=st.columns((5,4))
    with left_col:
        st.markdown("<h2 style='text-align:left; color:white;'>Method Used</h2>",True)
        st.markdown("""<p align="right"> <h3> 1. Data Analysis and Visualization <br> 2. Linear Regression <br> 3. Streamlit Framework <br> 4. Pandas <br> 5. Matplotlib <br> 6. Seaborn <br> 7.Random Forest Regression <br> 8.Ridge Regression <br> 9. Lasso Regression <br></h3></p>""",True)
    with right_col:
        st_lottie(strategy_animate,speed=1,loop=True,quality="low")

############################### OPTION 2 ########################################
elif nav=="DATA ANALYZE":
    st.markdown("<h1 style='text-align:center; color:white;'>ABOUT DATASET</h1>",True)
    st.markdown("""<p align="justify"; style="font-size:20px;">The data for this model is fairly simplified as it has very few missing pieces. 
            The raw data consists of a training dataset with the features listed above and their corresponding salaries. 
            Twenty percent of this training dataset was split into a test dataset with corresponding salaries.
            There is also a testing dataset that does not have any salary information available and was used as a substitute for real-world data.
            </p>
            """,True)
    
    st.write("---")
    
    st.markdown("<h1 style='text-align:center; color:white;'>ORIGINAL DATASET</h1>",True)
    st.write(df1)
    
    st.write("---")
    st.markdown("<h1 style='text-align:center; color:white;'>AFTER CHANGING IN DATASET</h1>",True)
    st.write("---")

    st.markdown("<h1 style='text-align:center; color:white;'>MODIFIED DATASET</h1>",True)
    st.write(df2)

    st.write("---")
    st.markdown("<h1 style='text-align:center; color:white;'>DATA VISUALIZATION</h1>",True)
    st.write("---")

    st.markdown("""<p align="justify"; style="font-size:20px;">In this part, We can easily visualize the data 
                    in the form of different types of charts. This will help us to easily understand the dataset
                    and its attributes relations within the datasets.
                    So, here below there are option to choose the graphs for visualize the data. Simply click on the name and you able to see graph.
                    </p>""",True)

    
    st.markdown("<h1 style='text-align:center; color:white;'>GRAPH</h1>",True)
    left_col,right_col=st.columns(2)
    with left_col:
        x_label=st.selectbox('CHOOSE THE X-AXIS LABEL',('yearsofexperience','yearsatcompany','gender',
                            'basesalary','Amazon','Apple','Facebook','Google','Microsoft',
                            'Business Analyst','Data Scientist', 'Hardware Engineer', 'Human Resources',
                            'Management Consultant', 'Marketing', 'Mechanical Engineer',
                            'Product Designer', 'Product Manager', 'Recruiter', 'Sales',
                            'Software Engineer', 'Software Engineering Manager',
                            'Solution Architect', 'Technical Program Manager'))
    with right_col:
        y_label=st.selectbox('CHOOSE THE Y-AXIS LABEL',('Amazon','Apple','Facebook','Google','Microsoft',
                            'yearsofexperience','yearsatcompany','gender','basesalary',
                            'Business Analyst','Data Scientist', 'Hardware Engineer', 'Human Resources',
                            'Management Consultant', 'Marketing', 'Mechanical Engineer',
                            'Product Designer', 'Product Manager', 'Recruiter', 'Sales',
                            'Software Engineer', 'Software Engineering Manager',
                            'Solution Architect', 'Technical Program Manager'))
    if (len(x_label)!=0 and len(y_label)!=0):    
        st.bar_chart(df2,x=x_label,y=y_label)
    
    st.write("---")

    st.markdown("<h1 style='text-align:center; color:white;'>Technologies & Libraries Used</h1>",True)
    left_col,right_col=st.columns(2)
    with left_col:
        st.markdown("""<p align="right"><h3> 1. pandas <br> 2. Numpy <br> 3. Scikit-Learn <br> 4. Jupyter <br> 5. Streamlit <br> 6. Python <br> 7. Word2Num</h3></p>""",True)
    with right_col:
        st_lottie(technology_animate,speed=1,loop=True,quality="low")
    
    st.write("---")
############################### OPTION 3 ########################################
elif nav=="PREDICT SALARY":
    st_lottie(welcome_animate,speed=1,loop=False,quality="low")
    name=st.text_input("Type your name")
    if len(name)==0:
        pass
    elif len(name)!=0:
        left_col,right_col=st.columns(2)
        with left_col:
            st_lottie(greet_animate,speed=1,loop=False,quality="low")
        with right_col:
            st.write("# Hi,",name)
            st.write("### Enter the following details to predict salary")

            yearofexperience=st.number_input("Years of Experience You Have:")
            yearatcompany=st.number_input("Years at Company:")
            gender=st.selectbox('Specify Your Gender',('Male','Female'))
            if gender=="Male":
                gender=1
            elif gender=="Female":
                gender=0
            
            amazon=0;apple=0;facebook=0;google=0;microsoft=0
            company=st.selectbox('Select Company',('amazon','apple','facebook','google','microsoft'))
            if company=='amazon':
                amazon=1
            elif company=='apple':
                amazon=1
            elif company=='facebook':
                facebook=1
            elif company=='google':
                google=1
            elif company=='microsoft':
                microsoft=1
            ba=0;ds=0;he=0;hr=0;mc=0;m=0;me=0;pd=0;pm=0;rec=0;sales=0;se=0;sem=0;sa=0;tpm=0
            profession=st.selectbox('Select Profession',('Business Analyst','Data Scientist', 'Hardware Engineer', 'Human Resources',
                                                         'Management Consultant', 'Marketing', 'Mechanical Engineer',
                                                         'Product Designer', 'Product Manager', 'Recruiter', 'Sales',
                                                         'Software Engineer', 'Software Engineering Manager',
                                                         'Solution Architect', 'Technical Program Manager'))
            
            if profession=="Business Analyst":
                ba=0
            elif profession=="Dat1a Scientist":
                ds=1
            elif profession=="Hardware Engineer":
                he=1
            elif profession=="Human Resources":
                hr=1
            elif profession=="Management Consultant":
                mc=1
            elif profession=="Marketing":
                m=1
            elif profession=="Mechanical Engineer":
                me=1
            elif profession=="Product Designer":
                pd=1
            elif profession=="Product Manager":
                pm=1
            elif profession=="Recruiter":
                rec=1
            elif profession=="Sales":
                saler=1
            elif profession=="Software Engineer":
                se=1
            elif profession=="Software Engineer Manager":
                sem=1
            elif profession=="Solution Architect":
                sa=1
            elif profession=="Technical Program Manager":
                tpm=1       

            feature=df2.drop(columns=['basesalary'])
            label=df2['basesalary']
            x_train,x_test,y_train,y_test=train_test_split(feature,label,test_size=0.2)
            option=st.selectbox('Choose the Model to Predict for Prediction',('Random Forest Regression','Linear Regression','Ridge Regression','Lasso Regression'))
            input_data=[[yearofexperience,yearatcompany,gender,amazon,apple,facebook,google,microsoft,ba,ds,he,hr,mc,m,me,pd,pm,rec,sales,se,sem,sa,tpm]]
            
            predict=st.button("Predict")

            if predict:
                if(int(yearatcompany)==0 and int(yearofexperience)==0):
                    st.error("All fields are Mandatory :grey_exclamation:")
                else:
                    if option=="Random Forest Regression":
                        model=RandomForestRegressor()
                        model.fit(x_train,y_train)
                        data=[[experience,test_score,interview_score]]
                        salary=int(model.predict(input_data))
                        st.write("### Predicted Salary is:",salary)
                    
                    elif option=="Linear Regression":
                        model=LinearRegression()
                        model.fit(x_train,y_train)
                        salary=int(model.predict(input_data))
                        st.write("### Predicted Salary is:",salary)
                    
                    elif option=="Ridge Regression":
                        model=Ridge()
                        model.fit(x_train,y_train)
                        salary=int(model.predict(input_data))
                        st.write("### Predicted Salary is:",salary)
                    
                    elif option=="Lasso Regression":
                        model=Lasso()
                        model.fit(x_train,y_train)
                        salary=int(model.predict(input_data))
                        st.write("### Predicted Salary is:",salary)








        
