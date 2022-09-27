# Import libraries from vsCode pip install
import folium
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn.datasets
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import plotly.figure_factory as ff
from streamlit_folium import st_folium
import folium

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set()
st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')
sns.set()
# Bước 2: Nhập dữ liệu đầu vào
#ex1 = 'ex1.xlsx'
#df = pd.read_excel(ex1, index_col=0)
# print(df.head()) # print the first 5 rows
data = pd.read_csv("house_price.csv")
st.write(data)

# data.corr() is used to find the pairwise correlation of all columns in the dataframe.
data.corr()
# A heatmap is a plot of rectangular data as a color-encoded matrix for corr() function
fig, ax = plt.subplots(figsize=(15, 15), dpi=600)
sns.heatmap(data.corr(), annot=True)
st.header("A heatmap for Boston House price prediction")
st.write(fig)
# boxplot is a method for graphically depicting groups of numerical data through their quartiles
fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(x='price', data=data, orient='h', width=0.8,
            fliersize=3, showmeans=True, ax=ax)
st.write(fig)
# Bước 5: Vẽ phương trình dự báo cho một biến
# phase 1: define the dependent variable and independent variables
y = data['price']
f = ["sqft_living"]

# phase 2: add constant
x2 = data[f]

x3 = sm.add_constant(x2)
x3.head()
# phase 4: plot and present linear equation
plt.figure(figsize=(15, 6), dpi=300)
plt.scatter(x2, y)
yhat = 280.6236*x3 - 4.358e+04
fig = plt.plot(x3, yhat, lw=4, c='red', label='regression line')
plt.xlabel('sqft_living', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()
#st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

# Boxplots to view to distribution of all continuous variables
# Using the date column to obtain the year that the house was sold
data['date'] = data['date'].astype('str')

for i in range(len(data.index)):
    data.loc[i, 'date'] = data.loc[i, 'date'][0:4]

data['date'] = data['date'].astype('int64')
# Renaming the column date as year_sold
data.rename(columns={'date': 'year_sold'}, inplace=True)

# If the sqft_living, sqft_living15 and sqft_lot, sqft_lot15 columns are not the same then it implies
# that the house has been renovated. A column renovated is created with 1 - renovated, 0 - not renovated
# data.loc[:,['sqft_living','sqft_lot','sqft_living15','sqft_lot15','yr_renovated']].head(10)
data['renovated'] = np.where((data['sqft_living'] != data['sqft_living15']) | (
    data['sqft_lot'] != data['sqft_lot15']), 1, 0)
# data.loc[:,['sqft_living','sqft_lot','sqft_living15','sqft_lot15','yr_renovated','renovated']].head(20)

# The yr_renovated column has mostly 0 values and we have obtained the renovation information, so it is dropped
# Columns id, sqft_living and sqft_lot won't be used, so they are dropped as well
data.drop(['id', 'sqft_living', 'sqft_lot',
          'yr_renovated'], axis=1, inplace=True)

# The age of the buidlding at the time it is sold is added as a new column
data['age_of_building'] = data['year_sold']-data['yr_built']

# Column yr_built is now  dropped since column age_of_building is created
data.drop('yr_built', axis=1, inplace=True)

# Columns bathrooms and floors have float values wihich is not possible, so they are floored
data['bathrooms'] = np.floor(data['bathrooms'])
data['floors'] = np.floor(data['floors'])

# Columns are changed to appropriate data types
data['waterfront'] = data['waterfront'].astype('category')
data['condition'] = data['condition'].astype('category')
data['grade'] = data['grade'].astype('category')
data['bathrooms'] = data['bathrooms'].astype('int64')
data['floors'] = data['floors'].astype('int64')
data['renovated'] = data['renovated'].astype('category')
data['zipcode'] = data['zipcode'].astype('category')
st.header("Boxplots to view to distribution of all continuous variables")
sns.set(rc={'figure.figsize': (15, 15)})
fig = plt.figure()

ax1 = fig.add_subplot(4, 3, 1)
sns.boxplot(y=data['price'], ax=ax1, width=0.3, color='turquoise')
ax1.set_title('Price of the houses')

ax2 = fig.add_subplot(4, 3, 2)
sns.boxplot(y=data['bedrooms'], ax=ax2, width=0.3, color='royalblue')
ax2.set_title('Number of bedrooms')

ax3 = fig.add_subplot(4, 3, 3)
sns.boxplot(y=data['bathrooms'], ax=ax3, width=0.3, color='cyan')
ax3.set_title('Number of bathrooms')

ax4 = fig.add_subplot(4, 3, 4)
sns.boxplot(y=data['floors'], ax=ax4, width=0.3, color='gold')
ax4.set_title('Number of floors')

ax5 = fig.add_subplot(4, 3, 5)
sns.boxplot(y=data['view'], ax=ax5, width=0.3, color='plum')
ax5.set_title('Number of times viewed')

ax6 = fig.add_subplot(4, 3, 6)
sns.boxplot(y=data['sqft_above'], ax=ax6, width=0.3, color='red')
ax6.set_title('Square footage of house apart from basement')

ax7 = fig.add_subplot(4, 3, 7)
sns.boxplot(y=data['sqft_basement'], ax=ax7, width=0.3, color='indigo')
ax7.set_title('Square footage of basement')

ax8 = fig.add_subplot(4, 3, 8)
sns.boxplot(y=data['sqft_living15'], ax=ax8, width=0.3, color='salmon')
ax8.set_title('Living room area')

ax9 = fig.add_subplot(4, 3, 9)
sns.boxplot(y=data['sqft_lot15'], ax=ax9, width=0.3, color='silver')
ax9.set_title('Lot size area')

ax10 = fig.add_subplot(4, 3, 10)
sns.boxplot(y=data['age_of_building'], ax=ax10,
            width=0.3, color='mediumaquamarine')
ax10.set_title('Age of buiding')

plt.show()
st.pyplot()

plt.figure(figsize=(12, 8))
data.groupby('zip_lat_long')['price'].mean().plot.bar()
plt.xlabel('Directions')
plt.ylabel('Average price')
plt.title('Average price by directions')
plt.show()
st.pyplot()
