#import required classes and packages
from keras_dgl.layers import GraphCNN #loading Graph Neural Network class
import keras.backend as K
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
import pickle
from keras.layers import Dense, Dropout, Activation, Flatten
import os
from sklearn.preprocessing import StandardScaler
from keras.layers import  MaxPooling2D
from keras.layers import Convolution2D
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'welcome'

#loading drug side effects
side_effects = pd.read_csv("Dataset/Interaction_information.csv")
side_effects

side_effects = side_effects.values

#function to get drug side effect description using ID
def getSideEffect(y_pred):
    label = "No Prediction"
    for i in range(len(side_effects)):
        if side_effects[i, 1].strip() == "DDI type "+str(y_pred).strip():
            label = side_effects[i,0]
            break
    return label  

    #loading and displaying two sides drug bank dataset
dataset = pd.read_csv("Dataset/twosides_drugbank.csv")
Y = dataset['type'].ravel()#get drug side effect labels
dataset

#convert drug smile string into training vector so graph nodes can be created by GNN 
dataset.drop(['type'], axis = 1,inplace=True)
dataset = dataset.values
X = []
for i in range(len(dataset)):#loop all smiles string and then convert to vector
    X.append(dataset[i,0]+" "+dataset[i,1]+" "+dataset[i,2]+" "+dataset[i,3])
#convert each smile string into numeric vector
tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
X = tfidf_vectorizer.fit_transform(X).toarray()
print("Generated Vector from Drug Smile stringr")
print(X)

#split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("Dataset Train & Test Split Details")
print("80% Text data used to train algorithms : "+str(X_train.shape[0]))
print("20% Text data used to train algorithms : "+str(X_test.shape[0]))

#define global variables to save accuracy and other metrics
accuracy = []
precision = []
recall = []
fscore = []


@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index1():
    return render_template('index.html', msg='')

@app.route('/AdminLogin', methods=['GET', 'POST'])
def AdminLogin():
    return render_template('AdminLogin.html', msg='')

@app.route('/AdminLoginAction', methods=['GET', 'POST'])
def AdminLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('AdminScreen.html', msg="Welcome "+user)
        else:
            return render_template('AdminLogin.html', msg="Invalid login details")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

def getModel():
    extension_model = load_model("model/extension_weights.hdf5")
    return extension_model

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        testData = pd.read_csv("Dataset/testData.csv")#read test data
        extension_model = getModel()
        testData = testData.values
        test = []
        for i in range(len(testData)):#create array of smile string
            test.append(testData[i,0]+" "+testData[i,1]+" "+testData[i,2]+" "+testData[i,3])
        test = tfidf_vectorizer.transform(test).toarray()#convert array of smile string into vector
        test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
        predict = extension_model.predict(test)#apply extension CNN2D model to predict side effects from given drug smile string vector
        output = ""
        for i in range(len(predict)):
            y_pred = np.argmax(predict[i])
            output += " Predicted Side Effect ===> "+getSideEffect(y_pred)+"<br/><br/>"             
        return render_template('AdminScreen.html', msg=output)

if __name__ == '__main__':
    app.run()
