"""
Streamlit Housing App Demo
    
Make sure to install Streamlit with `pip install streamlit`.

Run `streamlit hello` to get started!

To run this app:

1. cd into this directory
2. Run `streamlit run streamlit_app.py`
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# We begin with Parts 3, 6, and 7, but uncomment the code in each of the other parts and save to see how the Streamlit application updates in your browser.


### PART 1 - Agenda
#
st.write('''
## Welcome To Streamlit!
In this Streamlit app we will cover:

- Markdown
- Importing data
- Displaying dataframes
- Graphing
- Interactivity with buttons
- Mapping
- Making predictions with user input
''')


## PART 2 - Markdown Syntax
#
st.write(
'''
### Markdown Syntax
You can use Markdown syntax to style your text. For example,

## Main Title
### Subtitle
#### Header

**Bold Text**

*Italics*

Ordered List

1. Apples
2. Oranges
3. Bananas

[This is a link!](https://docs.streamlit.io/en/stable/getting_started.html)

'''
)


# PART 3 - Seattle House Prices Table

st.write(
'''
## Seattle House Prices
We can import data into our Streamlit app using pandas `read_csv` then display the resulting dataframe with `st.dataframe()`.

''')

data = pd.read_csv('out.csv')
#data = data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
st.dataframe(data)


## PART 4 - Graphing and Buttons
#
st.write(
'''
### Graphing and Buttons
Let's graph some of our data with matplotlib. We can also add buttons to add interactivity to our app.
'''
)

fig, ax = plt.subplots()

ax.hist(data['PRICE'])
ax.set_title('Distribution of House Prices in $100,000s')

show_graph = st.checkbox('Show Graph', value=True)

if show_graph:
     st.pyplot(fig)


## PART 5 - Mapping and Filtering Data
#
st.write(
'''
## Mapping and Filtering Data
We can also use Streamlit's built in mapping functionality.
Furthermore, we can use a slider to filter for houses within a particular price range.
'''
)

price_input = st.slider('House Price Filter', int(data['PRICE'].min()), int(data['PRICE'].max()), 500000 )

price_filter = data['PRICE'] < price_input
st.map(data.loc[price_filter, ['lat', 'lon']])


# PART 6 - Linear Regression Model

st.write(
'''
## Train a Linear Regression Model
Now let's create a model to predict a house's price from its square footage and number of bedrooms.
'''
) 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

clean_data = data.dropna(subset=['PRICE', 'SQUARE FEET', 'BEDS'])

X = clean_data[['SQUARE FEET', 'BEDS']]
y = clean_data['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y)

## Warning: Using the above code, the R^2 value will continue changing in the app. Remember this file is run upon every update! Set the random_state if you want consistent R^2 results.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

st.write(f'Test R²: {lr.score(X_test, y_test):.3f}')


# PART 7 - Predictions from User Input

st.write(
'''
## Model Predictions
And finally, we can make predictions with our trained model from user input.
'''
)

sqft = st.number_input('Square Footage of House', value=2000)
beds = st.number_input('Number of Bedrooms', value=3)

input_data = pd.DataFrame({'sqft': [sqft], 'beds': [beds]})
pred = lr.predict(input_data)[0]

st.write(
f'Predicted Sales Price of House: ${int(pred):,}'
)
"""
Streamlit Interactive Plots Demo
    
Example of a line chart of time-series simulation in Matplotlib
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Simulation")
st.write("'We have to realize that computers are simulators and then figure out what to simulate.' \n\n — Alan Kay")

trend = st.slider('Trend',  min_value=0.001, max_value=0.10, step=0.01)
noise = st.slider('Noise',min_value=0.01,  max_value=0.10, step=0.01)
st.write(f"Trend = {trend} \n\n Noise = {noise}")

intial_value = 1
n_series = 10 
time_series = np.cumprod(intial_value + np.random.normal(trend, noise, (100, n_series)), 
                         axis=0)

# st.line_chart(time_series)

fig, ax = plt.subplots()
for ts in time_series.T:
    ax.plot(ts)

st.pyplot(fig)

# Notes:
# - Switch from function to procedural
# - Lag in rendering
# - When you deploy you have to add an external dependency file (requirements.txt)

"""
Streamlit Sidebar Demo
    
Let's explore more advanced features of Streamlit:
https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py

Add widgets to sidebar
Move controls to sidebar 
"""

import pandas    as pd
import streamlit as st


st.title("Airline passenger data")
st.write("Plot airline passenger data over time")

url = "https://raw.githubusercontent.com/brianspiering/datasets/main/airline_passenger_data.csv"
df = pd.read_csv(url)

# Sidebar items
st.sidebar.markdown("# Controls") # Must be .markdown method, not .write method
st.sidebar.markdown("Adjust start index and number of months")
start_index = st.sidebar.slider('Start index',  min_value=0,
                        max_value=df.shape[0], step=1)
n_months = st.sidebar.slider('Number of months',  min_value=2,
                     max_value=df.shape[0]-start_index, step=1)

# Data selection
data = df.iloc[start_index:start_index+n_months, 2]

# Data visualization
st.line_chart(data)

"""
Streamlit & Scikit-learn Demo

Inspired by https://github.com/woosal1337
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn                 import datasets
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score
from sklearn.decomposition   import PCA
from sklearn.model_selection import train_test_split
import streamlit as st

st.title("Streamlit Example")

st.write('''
# Explore different classifiers
''')

st.sidebar.markdown("Select from the menus below:")
dataset_name = st.sidebar.selectbox("Select Dataset", 
                                   ("Iris Dataset", "Breast Cancer Dataset", "Wine Dataset"))
classifier_name = st.sidebar.selectbox("Select Classifier", 
                                      ("KNN", "SVM", "Random Forest"))


def load_dataset(dataset_name):
    if dataset_name.lower() == "iris dataset":
        data = datasets.load_iris()
    elif dataset_name.lower() == "breast cancer dataset":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X, y = data.data, data.target
    return X, y


X, y = load_dataset(dataset_name)
st.write(f"Shape of the dataset: {X.shape}")
st.write(f"Number of classes: {len(np.unique(y))}")


def add_parameter_ui(clf_name):
    params = {}
    if clf_name.lower() == "knn":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name.lower() == "svm":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("C", 2, 15)
        params["max_depth"] = max_depth

        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators

    return params


params = add_parameter_ui(classifier_name)


def load_classifier(clf_name, params):
    if clf_name.lower() == "knn":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name.lower() == "svm":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"],
                                     random_state=1234)

    return clf

clf = load_classifier(clf_name=classifier_name, params=params)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(x_test)

# Displaying the accuracy and the model details
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier Name: {classifier_name}")
st.write(f"Accuracy: {acc:.2%}")

# Plotting
pca = PCA(n_components=2) # Reduce number of dimensions to visualize in 2d
X_projected = pca.fit_transform(X)

x1, x2 = X_projected[:, 0], X_projected[:, 1]
 
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)