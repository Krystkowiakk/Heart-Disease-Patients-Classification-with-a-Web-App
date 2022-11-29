import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
from PIL import Image


image = Image.open('streamlit_app/i_1.png')
st.image(image)

st.title('Heart Disease Indicators')
st.caption('From Behavioural Risk Factor Surveillance System dataset.')

show_data = st.checkbox('Show Raw Data', value=False)

#st.text('Loading...')
data = pd.read_csv('streamlit_app/out.csv')
#st.text('Loading...Done')

if show_data:
    st.subheader('Raw Data')
    st.caption('The dataset originally comes from the CDC and is a major part of the Behavioural Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to gather data on the health status of U.S. residents. BRFSS completes more than 400,000 adult interviews each year, making it the largest continuously conducted health survey system in the world.". The most recent dataset (as of February 15, 2022) includes data from 2020.')
    st.dataframe(data.drop(columns=['Unnamed: 0']))

#hist_values = np.histogram(data['HeartDisease'], bins = 2)[0]
#st.bar_chart(hist_values)

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
ax.set(ylabel='percent', xlabel='x')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for i, p in enumerate(ax.patches):
    percentage = '{:.1f}%'.format(s[i])
    x = p.get_x() + 0.4
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center')
st.pyplot(fig2)


st.subheader('Heart Disease vs Diferent Features')
st.caption('How heart disease is releated to deferent features from dataset?')

feature = st.selectbox(
    'Choose:',
    ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Race'))
fig, ax1 = plt.subplots(figsize=(10, 4))
graph = sns.countplot(ax=ax1,x = feature , data = data, hue='HeartDisease')
#graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = int(p.get_height())
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
st.pyplot(fig)

st.subheader('Heart Disease vs Age & Lifestyle') #change to diferent plot, keeep age, alco and smoke as percentage
st.caption('And how heart disease is releated to age and lifestyle?')
show_numbers = st.checkbox('Show numbers', value=False)  #move that switch on side
fig, ax1 = plt.subplots(figsize=(10, 4))

show_smokers = st.radio(
    "How smoking influence the Heart Disease? Show:",
    ('Both', 'Smoking', 'No Smoking'))

show_alcohol = st.radio(
    "How alcohol drinking influence the Heart Disease? Show:",
    ('Both', 'Alcohol Drinking', 'No Alcohol Drinking'))    

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

graph = sns.countplot(ax=ax1,x = 'AgeCategory' , data = data_filtered, hue='HeartDisease', order=['18-24', '25-29', '30-34', '35-39','40-44', '45-49', '50-54', '55-59', '60-64','65-69', '70-74', '75-79', '80 or older'])
if show_numbers:
    for p in graph.patches:
        height = int(p.get_height())
        graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
st.pyplot(fig)




