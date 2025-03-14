import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("E:\Project self\VS\display.csv")
df.head()
df.describe()
df.info()
df.isnull().sum()
for columns in df.columns:
  print("{}-{}".format(columns,df[columns].unique()))
df['Cost_Per_Person'].unique()
df.columns
df1=df.drop(["Name","Type"],axis=1)
df1.columns
col_list=['Afghani', 'African', 'American', 'Andhra', 'Arabian',
       'Asian', 'Assamese', 'Awadhi', 'BBQ', 'Bakery', 'Belgian', 'Bengali',
       'Beverages', 'Bihari', 'Bohri', 'British', 'Burmese', 'Cantonese',
       'Chettinad', 'Chinese', 'Continental', 'Desserts', 'European',
       'Fast Food', 'French', 'German', 'Goan', 'Greek', 'Gujarati',
       'Healthy Food', 'Hyderabadi', 'Indonesian', 'Iranian', 'Italian',
       'Japanese', 'Jewish', 'Kashmiri', 'Kerala', 'Konkan', 'Korean',
       'Lebanese', 'Lucknowi', 'Maharashtrian', 'Malaysian', 'Mangalorean',
       'Mediterranean', 'Mexican', 'Middle Eastern', 'Modern Indian',
       'Mughlai', 'Naga', 'Nepalese', 'North Eastern', 'North Indian', 'Oriya',
       'Parsi', 'Portuguese', 'Rajasthani', 'Russian', 'Seafood', 'Sindhi',
       'Singaporean', 'South American', 'South Indian', 'Spanish',
       'Sri Lankan', 'Tamil', 'Thai', 'Tibetan', 'Turkish', 'Vegan',
       'Vietnamese']
df1.drop(col_list,axis=1,inplace=True)

# barplot(data=df1, x = 'Rating', y='Cost_Per_Person')
# plt.xticks(rotation=90)
# plt.show()
# sns.heatmap(df1[['Cost_Per_Person', 'Rating']].corr(),annot=True)
# sns.barplot(data=df1, x = 'City', y='Rating')
# plt.xticks(rotation=90)
# plt.show()
df1['Menu'].replace({'Yes':1, 'No':0})
df1.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(df1['Menu'])
df1['Menu']= label


print(label)
df1.shape
df1.head()
label = le.fit_transform(df1['Delivery'])
df1['Delivery']= label
df1.head()
label = le.fit_transform(df1['Booking'])
df1['Booking']= label
df_Category_dummies = pd.get_dummies(df1['Category'], dummy_na=False).astype('float64')
df1 = pd.concat([df1,df_Category_dummies], axis=1)
df1=df1.drop('Category', axis=1)
a=df1['Price_Category'].replace({'Expensive':3,'Affordable':2, 'Resonable':1, 'Cheap':0})
df1.drop('Price_Category', axis=1)
df1['Price_Category']=a
df_City_dummies = pd.get_dummies(df1['City'], dummy_na=False).astype('float64')
df1 = pd.concat([df1,df_City_dummies], axis=1)
df1=df1.drop('City',axis=1)
df1.head()


df1.head()
df1['Price_Category'].unique()
for columns in df1.columns:
  print("{}-{}".format(columns,df1[columns].dtypes))
df1['Cost_Per_Person']=df1['Cost_Per_Person'].astype(int)
df1.head()
df1.isnull().sum()
df1['Rating'].isnull().sum()
df1['Cost_Per_Person'].dtypes
df1.shape
X = df1.drop('Rating',axis=1)
Y = df1['Rating']
X.shape, Y.shape

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,
                                                 random_state=7)
X_train.shape, Y_train.shape , X_test.shape, Y_test.shape
from sklearn.linear_model import LinearRegression, Lasso, Ridge
lr = LinearRegression() #OLS
lr.fit(X_train,Y_train)
Y_pred = lr.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score, accuracy_score
print( r2_score(Y_test,Y_pred))
X_test.head()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rf = RandomForestRegressor(n_estimators=100, random_state=7)
rf.fit(X_train, Y_train)  # Train the model
y_pred = rf.predict(X_test)  # Make predictions

# Evaluate the model
print("R² Score:", r2_score(Y_test, y_pred))
import pickle
pickle.dump(rf,open('Model2.pkl','wb'))
Model =pickle.load(open('Model2.pkl','rb'))
