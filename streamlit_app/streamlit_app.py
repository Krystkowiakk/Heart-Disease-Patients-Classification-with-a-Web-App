import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
#import numpy as np
import seaborn as sns
from PIL import Image
import pickle
#from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mtick

#data loading
data = pd.read_csv('streamlit_app/out.csv')

#dummy variables for categorical features generation
def dum_gen(col, lis):
    for l in lis:
        if df_check[col][0] == l:
            df_check[l] = 1
        else:
            df_check[l] = 0

#sidebar form for user input and prediction
with st.sidebar.form("my_form"):
   st.write("Check if you should visit the doctor!")
   Sex = st.selectbox("Sex", ("Female", "Male"))
   AgeCategory = st.selectbox("Age Category", ('AgeCategory_18-24', 'AgeCategory_25-29',
       'AgeCategory_30-34', 'AgeCategory_35-39', 'AgeCategory_40-44',
       'AgeCategory_45-49', 'AgeCategory_50-54', 'AgeCategory_55-59',
       'AgeCategory_60-64', 'AgeCategory_65-69', 'AgeCategory_70-74',
       'AgeCategory_75-79', 'AgeCategory_80 or older')) #names to be fixed
   Race = st.selectbox("Race", ('Race_Hispanic', 'Race_White', 'Race_Black', 'Race_American Indian/Alaskan Native', 'Race_Asian',  'Race_Other', ))
   f_height = st.slider('Height? (cm)', 50.0, 230.0, (160.0), step=1.0) #check standard measures
   f_weight = st.slider('Weight? (kg)', 35.0, 200.0, (65.0), step=1.0)  #check standard measures
   BMI = 10000*f_weight/(f_height)**2
   GenHealth = st.selectbox("General Health", ('GenHealth_Excellent','GenHealth_Very good', 'GenHealth_Good','GenHealth_Fair', 'GenHealth_Poor' ))
   PhysicalActivity = st.selectbox("Phisical Activity", ("Yes", "No"))
   SleepTime = st.slider('Sleep Time?', 1.0, 24.0, (7.0), step=1.0)  #check standard measures
   Smoking = int(st.checkbox('Smoking'))
   AlcoholDrinking = int(st.checkbox('Alcohol Drinking'))
   Stroke = int(st.checkbox('Stroke'))
   DiffWalking = int(st.checkbox('Difficulties with Walking'))
   Asthma = int(st.checkbox('Asthma'))
   KidneyDisease = int(st.checkbox('Kidney Disease'))
   SkinCancer = int(st.checkbox('Skin Cancer'))
   Diabetic = st.selectbox("Diabetic", ('Diabetic_No', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes', 'Diabetic_Yes (during pregnancy)'))
   submitted = st.form_submit_button("Submit")
   if submitted:
        input_data = {
            'BMI': [BMI],
            'Smoking': [Smoking],
            'AlcoholDrinking': [AlcoholDrinking],
            'Stroke': [Stroke],
            'DiffWalking': [DiffWalking],
            'Sex': [Sex],
            'PhysicalActivity': [PhysicalActivity],
            'SleepTime': [SleepTime],
            'Asthma': [Asthma],
            'KidneyDisease': [KidneyDisease],
            'SkinCancer': [SkinCancer],
            'AgeCategory': [AgeCategory],
            'Race': [Race],
            'Diabetic': [Diabetic],
            'GenHealth': [GenHealth]
        }
        df_check = pd.DataFrame(data = input_data)
        dum_gen('AgeCategory', ['AgeCategory_18-24', 'AgeCategory_25-29', 'AgeCategory_30-34', 'AgeCategory_35-39', 'AgeCategory_40-44', 'AgeCategory_45-49', 'AgeCategory_50-54', 'AgeCategory_55-59', 'AgeCategory_60-64', 'AgeCategory_65-69', 'AgeCategory_70-74', 'AgeCategory_75-79', 'AgeCategory_80 or older'])
        dum_gen('Race', ['Race_American Indian/Alaskan Native', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White'])
        dum_gen('Diabetic', ['Diabetic_No', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes', 'Diabetic_Yes (during pregnancy)'])
        dum_gen('GenHealth',['GenHealth_Excellent','GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good'])
        df_check = df_check.drop(columns=['AgeCategory','Race','Diabetic', 'GenHealth'])
        df_check =  df_check.replace({'Yes':1, 'No':0, 'Female':1,'Male':0 })
        scalar = pickle.load(open('streamlit_app/scaled.pkl', 'rb'))
        df_check_scal = scalar.transform(df_check)
        model = pickle.load(open('streamlit_app/model.pkl', 'rb'))
        prediction = model.predict_proba(df_check_scal)
        if prediction[0][1] >= 0.08: #treshold adjusted during model tuning
            st.title("Better visit the doctor!")
        else:
            st.title("Seems like you are fine")
        st.write("Remember! That app is not created by the doctor but if prediction concerns you, maybe you should visit one.")
image = Image.open('streamlit_app/i_1.png')
st.image(image)

st.title('Heart Disease Indicators')
st.caption('From Behavioural Risk Factor Surveillance System dataset.')

# heart disease and its relation to different features chart
st.subheader('Heart Disease and its relation to different features')
feature = st.selectbox(
   'This chart shows how likely a person is to have heart disease based on different characteristics. It helps us understand which factors may affect heart disease risk. By looking at this data, we can find patterns or risk factors that can help prevent or treat heart disease.',
   ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic', 'GenHealth', 'Race'))
fig, ax = plt.subplots(figsize=(10, 4))
# create a new dataframe to calculate the percentages
unique_values = data[feature].unique()
df = pd.DataFrame(columns=[feature, 'HeartDisease'])
for value in unique_values:
    sub_data = data[data[feature] == value]
    count = len(sub_data)
    hd_count = len(sub_data[sub_data['HeartDisease'] == 'Yes'])
    percent = (hd_count / count) * 100
    df = df.append({feature: value, 'HeartDisease': percent}, ignore_index=True)
# create the bar plot
sns.barplot(x=feature, y='HeartDisease', data=df, ax=ax)
a = ax.get_xticklabels()
for tick in a:
    if tick.get_text() == 'American Indian/Alaskan Native':
        tick.set_text('Amer.Indian/Alaskan')
ax.set_xticklabels(a)
# format the y-axis to show percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
st.pyplot(fig)


st.subheader('Heart Disease vs Age & Lifestyle') #change to diferent plot, keeep age, alco and smoke as percentage
st.caption('And how heart disease is releated to age and lifestyle?')

col1, col2= st.columns(2)

with col1:
    show_smokers = st.checkbox("Smoking", value=False)

with col2:
    show_alcohol = st.checkbox("Alcohol Drinking", value=False)

if show_smokers:
    data_filtered = data[data['Smoking']=='Yes']
else:
    data_filtered = data[data['Smoking']=='No']

if show_alcohol:
    data_filtered = data_filtered[data_filtered['AlcoholDrinking']=='Yes']
else:
    data_filtered = data_filtered[data_filtered['AlcoholDrinking']=='No']

fig, axs = plt.subplots(4,3, figsize=(15,10), sharey=True)
plt.subplots_adjust(hspace = 0.3)
i = 0
j = 0
for age in data_filtered['AgeCategory'].unique():
    data_age = data_filtered[data_filtered['AgeCategory'] == age]
    risk_of_heart_disease = (data_age['HeartDisease'].value_counts()[1] / len(data_age))*100
    axs[i,j].bar(['Heart Disease Risk'], risk_of_heart_disease)
    axs[i,j].set_title(age)
    if j == 2:
        i += 1
        j = 0
    else:
        j += 1

st.pyplot(fig)



#bottom part checkbox showing raw data and target distribution
show_data = st.checkbox('More about Data, Target Distribution &  Raw Data', value=False)
if show_data:
    st.caption('The dataset originally comes from the CDC and is a major part of the Behavioural Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to gather data on the health status of U.S. residents. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.". The most recent dataset (as of February 15, 2022) includes data from 2020.')
    st.subheader('Target Distribution')
    st.caption('The dataset is unbalanced as there is more healthy people than ones having heart disease.')
    s = data['HeartDisease'].value_counts(normalize=True, sort=False).mul(100)
    fig = plt.figure(figsize = (10,4))
    ax = sns.barplot(x=s.index, y=s)
    ax.set(ylabel="", xlabel='HeartDisease')
    ax.set(yticklabels=[])
    for i, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(s[i])
        x = p.get_x() + 0.4
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center')
    st.pyplot(fig)
    st.subheader('Raw Data')
    st.dataframe(data.drop(columns=['Unnamed: 0']))


