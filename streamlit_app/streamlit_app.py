import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
from PIL import Image

#st.text('Loading...')
data = pd.read_csv('streamlit_app/out.csv')
#st.text('Loading...Done')


with st.sidebar.form("my_form"):
   st.write("Check if you should visit the doctor yourself!")
   Sex = st.selectbox("Sex", ("Female", "Male"))
   AgeCategory = st.selectbox("Age Category", ('18-24', '25-29', '30-34', '35-39','40-44', '45-49', '50-54', '55-59', '60-64','65-69', '70-74', '75-79', '80 or older'))
   Race = st.selectbox("Race", ('White', 'Asian', 'Hispanic', 'Black', 'American Indian/Alaskan Native', 'Other'))
   f_height = st.slider('Height?', 50.0, 230.0, (160.0), step=1.0) #check standard measures
   f_weight = st.slider('Weight?', 35.0, 200.0, (65.0), step=1.0)  #check standard measures
   BMI = 10000*f_weight/(f_height)**2
   GenHealth = st.selectbox("General Health", ('Excellent', 'Very good', 'Good', 'Fair', 'Poor'))
   PhysicalActivity = st.selectbox("Phisical Activity", ("Yes", "No"))
   SleepTime = st.slider('Sleep Time?', 1.0, 24.0, (7.0), step=1.0)  #check standard measures
   Smoking = st.checkbox('Smoking')
   AlcoholDrinking = st.checkbox('Alcohol Drinking')
   Stroke = st.checkbox('Stroke')
   #PhysicalHealth and MentalHealth -> exclude as dificult to express and not a strong features?
   DiffWalking = st.checkbox('Difficulties with Walking')
   Asthma = st.checkbox('Asthma')
   KidneyDisease = st.checkbox('Kidney Disease')
   SkinCancer = st.checkbox('Skin Cancer')  
   Diabetic = st.selectbox("Diabetic", options=(i for i in data.Diabetic.unique()))
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("BMI", Sex, "checkbox", AgeCategory)

image = Image.open('streamlit_app/i_1.png')
st.image(image)

st.title('Heart Disease Indicators')
st.caption('From Behavioural Risk Factor Surveillance System dataset.')

show_data = st.checkbox('Show Raw Data & Target Distribution', value=False)



if show_data:
    st.subheader('Raw Data')
    st.caption('The dataset originally comes from the CDC and is a major part of the Behavioural Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to gather data on the health status of U.S. residents. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.". The most recent dataset (as of February 15, 2022) includes data from 2020.')
    st.dataframe(data.drop(columns=['Unnamed: 0']))
    st.subheader('Target Distibution')
    st.caption('The dataset is unbalaced as we there is more healthy people than ones having heart desease.') #better as pieplot?

    #fig = plt.figure(figsize=(10, 4))
    #graph = sns.countplot(x='HeartDisease', data=data)
    #for p in graph.patches:
    #    height = int(p.get_height())
    #    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
    #st.pyplot(fig)

    import matplotlib.ticker as mtick
    s = data['HeartDisease'].value_counts(normalize=True, sort=False).mul(100)
    fig2 = plt.figure(figsize = (10,4))
    ax = sns.barplot(x=s.index, y=s)
    ax.set(ylabel="", xlabel='HeartDisease')
    #ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set(yticklabels=[])
    for i, p in enumerate(ax.patches):
        percentage = '{:.1f}%'.format(s[i])
        x = p.get_x() + 0.4
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center')
    st.pyplot(fig2)
#hist_values = np.histogram(data['HeartDisease'], bins = 2)[0]
#st.bar_chart(hist_values)





st.subheader('Heart Disease vs Diferent Features')

feature = st.selectbox(
   'How heart disease is releated to deferent features from dataset?',
   ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Race'))
fig, ax1 = plt.subplots(figsize=(10, 4))
total = float(len(data))
graph = sns.countplot(ax=ax1,x = feature , data = data, hue='HeartDisease')
graph.set(ylabel="")
graph.set(yticklabels=[])
#graph.yaxis.set_major_formatter(mtick.PercentFormatter())
for p in graph.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2
    y = p.get_height()
    graph.annotate(percentage, (x, y),ha='center')
st.pyplot(fig)

# feature = st.selectbox(
#    'How heart disease is releated to deferent features from dataset?',
#    ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Race'))
# fig, ax1 = plt.subplots(figsize=(10, 4))
# graph = sns.countplot(ax=ax1,x = feature , data = data, hue='HeartDisease')
# ##graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
# for p in graph.patches:
#   height = int(p.get_height())
#   graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
# st.pyplot(fig)

st.subheader('Heart Disease vs Age & Lifestyle') #change to diferent plot, keeep age, alco and smoke as percentage
st.caption('And how heart disease is releated to age and lifestyle?')

col1, col2, col3 = st.columns(3)

with col1:
    show_smokers = st.radio(
        "How smoking influence the Heart Disease? Show:",
        ('Both', 'Smoking', 'No Smoking'))

with col2:
    show_alcohol = st.radio(
        "How alcohol drinking influence the Heart Disease? Show:",
        ('Both', 'Alcohol Drinking', 'No Alcohol Drinking'))    

with col3:
    show_numbers = st.checkbox('Show numbers', value=False)  #move that switch on side

if show_smokers == 'Smoking':
    data_filtered = data[data['Smoking']=='Yes']
elif show_smokers == 'No Smoking':
    data_filtered = data[data['Smoking']=='No']
else:
    data_filtered = data

if show_alcohol == 'Alcohol Drinking':
    data_filtered = data_filtered[data_filtered['AlcoholDrinking']=='Yes']
elif show_alcohol == 'No Alcohol Drinking':
    data_filtered = data_filtered[data_filtered['AlcoholDrinking']=='No']
else:
    data_filtered = data_filtered

fig, ax1 = plt.subplots(figsize=(10, 4))
graph = sns.countplot(ax=ax1,x = 'AgeCategory' , data = data_filtered, hue='HeartDisease', order=['18-24', '25-29', '30-34', '35-39','40-44', '45-49', '50-54', '55-59', '60-64','65-69', '70-74', '75-79', '80 or older'])
graph.set(ylabel="")
graph.set(yticklabels=[])
if show_numbers:
    total = float(len(data_filtered))
    for p in graph.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()/1.3
        y = p.get_height()
        graph.annotate(percentage, (x, y),ha='center')
st.pyplot(fig)




