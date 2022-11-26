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
st.caption('The dataset is unbalaced as we there is more healthy people than ones having heart desease.')

fig = plt.figure(figsize=(10, 4))
graph = sns.countplot(x='HeartDisease', data=data)
for p in graph.patches:
    height = int(p.get_height())
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
st.pyplot(fig)

st.subheader('Heart Disease vs Diferent Features')
st.caption('How heart disease is releated to deferent features from dataset?')

feature = st.selectbox(
    'Choose:',
    ('Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer'))
fig, ax1 = plt.subplots(figsize=(10, 4))
graph = sns.countplot(ax=ax1,x = feature , data = data, hue='HeartDisease')
#graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = int(p.get_height())
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
st.pyplot(fig)

st.subheader('Heart Disease vs Age')
st.caption('And how heart disease is releated to age?')
show_numbers = st.checkbox('Show numbers', value=False)  #move that switch on side
# add buttons for smokers or drinkers only
fig, ax1 = plt.subplots(figsize=(10, 4))
graph = sns.countplot(ax=ax1,x = 'AgeCategory' , data = data, hue='HeartDisease')
if show_numbers:
    for p in graph.patches:
        height = int(p.get_height())
        graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
st.pyplot(fig)

#categorical_features=['Race', 'Diabetic', 'PhysicalActivity', 'GenHealth']



# PART 3 - Seattle House Prices Table


# st.write(
# '''
# ## Seattle House Prices
# We can import data into our Streamlit app using pandas `read_csv` then display the resulting dataframe with `st.dataframe()`.

# ''')


#data = data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})




# ## PART 5 - Mapping and Filtering Data
# #
# st.write(
# '''
# ## Mapping and Filtering Data
# We can also use Streamlit's built in mapping functionality.
# Furthermore, we can use a slider to filter for houses within a particular price range.
# '''
# )

# price_input = st.slider('House Price Filter', int(data['PRICE'].min()), int(data['PRICE'].max()), 500000 )

# price_filter = data['PRICE'] < price_input
# st.map(data.loc[price_filter, ['lat', 'lon']])

