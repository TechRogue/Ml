import pandas as pd 
import numpy as np 

#Splits your dataset into training and testing parts.
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as pit 
import seaborn as sns 

data = pd.read_csv("tested.csv")
data.info()
print(data.isnull().sum())

def preprocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Embarked"], inplace=True)
    
    #convert gender into 1,0
    df['Sex'] =  df['Sex'].map({'male':1, "female":0})
