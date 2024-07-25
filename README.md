# Credit-Card-Fraud-Detection

# Description

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation.
Due to confidentiality issues, there are not provided the original features and more background information about the data.
Features V1, V2, ... V28 are the principal components obtained with PCA;
The only features which have not been transformed with PCA are Time and Amount. Feature Time contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature Amount is the transaction Amount, this feature can be used for example-dependant cost-senstive learning.
Feature Class is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# import the necessary packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Loading Dataset 

data = pd.read_csv('creditcard.csv')
pd.options.display.max_columns = None

#  Display Top 5 Rows of The Dataset
data.head()

# Check Last 5 Rows of The Dataset
data.tail()

# Find Shape of Our Dataset (Number of Rows And Number of Columns)
data.shape

print ('number of rows',data.shape[0]),
print ('number of columns',data.shape[1])
data.value_counts('Class')

# Find Shape of Our Dataset (Number of Rows And Number of Columns)
data.info()

#  Calculate summary statistics
summary_statistics = data.describe()
summary_statistics

# Check Null Values In The Dataset
data.isnull().sum()
data.head()

# Creating subplots
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Histogram for 'Time' feature
sns.histplot(data['Time'], bins=50, ax=ax[0], kde=True)
ax[0].set_title('Distribution of Transaction Time')
ax[0].set_xlabel('Time (seconds)')
ax[0].set_ylabel('Frequency')

# Histogram for 'Amount' feature
sns.histplot(data['Amount'], bins=50, ax=ax[1], kde=True)
ax[1].set_title('Distribution of Transaction Amount')
ax[1].set_xlabel('Amount')
ax[1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

# The histograms for the 'Time' and 'Amount' features show the following:
Transaction Time: The distribution of transaction times shows some periodic patterns, which could be related to daily or weekly cycles in transaction frequency. This is common in financial transaction data, where activity can vary significantly depending on the time of day or week.
Transaction Amount: Most transactions are of lower amounts, with the frequency rapidly decreasing as the amount increases. This long-tailed distribution is typical in financial datasets, where small transactions are common and large transactions are relatively rare.
The 'Amount' feature shows a significant range and might need scaling or normalization for certain types of analysis or modeling.

# Box Plot

# Creating subplots for box plots
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# Box plot for 'Amount' by 'Class'
sns.boxplot(x='Class', y='Amount', data=data, ax=ax[0])
ax[0].set_title('Box Plot of Transaction Amount by Class')
ax[0].set_yscale('log')  # Using log scale due to wide range of 'Amount'
ax[0].set_xlabel('Class')
ax[0].set_ylabel('Amount (log scale)')

# Box plot for 'Time' by 'Class'
sns.boxplot(x='Class', y='Time', data=data, ax=ax[1])
ax[1].set_title('Box Plot of Transaction Time by Class')
ax[1].set_xlabel('Class')
ax[1].set_ylabel('Time (seconds)')

# Adjust layout
plt.tight_layout()
plt.show()

# The box plots provide insights into how the 'Amount' and 'Time' features vary between fraudulent and non-fraudulent transactions:

Transaction Amount by Class:
For non-fraudulent transactions (Class 0), the 'Amount' tends to be smaller and has a narrower interquartile range.
Fraudulent transactions (Class 1) show a wider range of 'Amount', with some outliers indicating very high transaction amounts. However, the median is still relatively low, suggesting that many fraudulent transactions are of lower amounts.
Transaction Time by Class:
The distribution of transaction times for both classes appears quite similar.
There's no immediately apparent difference in the distribution of transaction times between fraudulent and non-fraudulent transactions.

data.shape

# Checking and Removing Duplicates Values
data.duplicated().any()
data=data.drop_duplicates()
data.shape

#  Imbalanced Data
data['Class'].value_counts()
import seaborn as sns
sns.countplot(x='Class', data=data)

#  A few observations:
The dataset is likely highly imbalanced, with a much higher proportion of non-fraudulent transactions compared to fraudulent ones.
The V1-V28 features are the result of PCA, so they are already transformed and standardized

# Calculating the correlation matrix
corr_matrix = data.corr()

# Creating a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".1f")
plt.title('Correlation Heatmap of Features')
plt.show()

# The correlation heatmap provides a visual representation of the relationships between different features in the dataset:
Most of the V1-V28 features, which are results of PCA, show little to no correlation with each other. This is expected, as one of the purposes of PCA is to orthogonalize the components.
The 'Class' variable shows varying degrees of correlation with some of the V features, which indicates that certain features might be more informative in distinguishing fraudulent transactions.
The 'Time' and 'Amount' features do not exhibit strong correlations with the other features or with the 'Class' variable, suggesting that they might not be as discriminative on their own for identifying fraud.

# Data preprocessing using StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))
data.head()
data = data.drop(['Time'],axis=1)
data.head()
data.shape

# store feature matrix in X and response(Target) vector in y
X = data.drop(['Class'],axis=1)
y = data['Class']

# Splitting Datasets into the Training and Testing set and Test sets 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# checking the model before balancing the data
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred = log.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)

# Handeling Imbalanced Dataset With
*undersampling
*oversampling

# Undersampling
normal = data[data['Class']==0]
fraud = data[data['Class']==1]
normal.shape
fraud.shape
normal_sample = normal.sample(n=473)
normal_sample.shape
new_data = pd.concat([normal_sample,fraud],ignore_index=True)
new_data['Class'].value_counts()
new_data.head()
X = new_data.drop(['Class'],axis=1)
y = new_data['Class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred1 = log.predict(X_test)

from sklearn.metrics import accuracy_score

training_data_test = log.predict(X_train)

accuracy_score(y_train,training_data_test)

accuracy_score(y_test,y_pred1)

from sklearn.metrics import precision_score,recall_score,f1_score

precision_score(y_test,y_pred1)

recall_score(y_test,y_pred1)

f1_score(y_test,y_pred1)

# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

# Initialize the DecisionTreeClassifier with max_depth
dt_pruned = DecisionTreeClassifier(max_depth=5)  # You can adjust the max_depth value
# Fit the model
dt_pruned.fit(X_train, y_train)

y_pred_pruned = dt_pruned.predict(X_test)

training_data_test1 = log.predict(X_train)

accuracy_score(y_train,training_data_test1)

accuracy_score(y_test,y_pred_pruned)

precision_score(y_test,y_pred_pruned)

recall_score(y_test,y_pred_pruned)

f1_score(y_test,y_pred_pruned)

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred3 = rf.predict(X_test)

training_data_test2 = log.predict(X_train)

accuracy_score(y_train,training_data_test2)

accuracy_score(y_test,y_pred3)

precision_score(y_test,y_pred3)

recall_score(y_test,y_pred3)

f1_score(y_test,y_pred3)

final_data = pd.DataFrame({'Models':['LR','DT','RF'],'ACC':[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred_pruned)*100,accuracy_score(y_test,y_pred3)*100]})

final_data

sns.barplot(x='Models', y='ACC', data=final_data)

# Oversampling

X = data.drop(['Class'],axis=1)
y = data['Class']

X.shape

y.shape

from imblearn.over_sampling import SMOTE

X_res,y_res = SMOTE().fit_resample(X,y)

y_res.value_counts()

X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,random_state=42)

# LogisticRegression 

log = LogisticRegression()

log.fit(X_train,y_train)

y_pred1 = log.predict(X_test)

acc_train_data = log.predict(X_train)

accuracy_score(y_train,acc_train_data)

accuracy_score(y_test,y_pred1)

precision_score(y_test,y_pred1)

recall_score(y_test,y_pred1)

f1_score(y_test,y_pred1)



# Decision Tree Classifier 

from sklearn.tree import DecisionTreeClassifier

ds = DecisionTreeClassifier(max_depth=5)

ds.fit(X_train,y_train)

y_pred2 = ds.predict(X_test)

acc_train_data1 = log.predict(X_train)

accuracy_score(y_train,acc_train_data1)

accuracy_score(y_test,y_pred2)

precision_score(y_test,y_pred2)

recall_score(y_test,y_pred2)

f1_score(y_test,y_pred2)

# RandomForestClassifier 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rc = RandomForestClassifier()

rc.fit(X_train,y_train) 

 y_pred3 = rc.predict(X_test)

acc_train_data2 = log.predict(X_train)

accuracy_score(y_train,acc_train_data2)

accuracy_score(y_test,y_pred3)

precision_score(y_test,y_pred3)

recall_score(y_test,y_pred3)

f1_score(y_test,y_pred3)

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred3)

# Print the AUC-ROC score
print(f"AUC-ROC Score: {auc_roc}")

final_data = pd.DataFrame({'Models':['LR','DT','RF'],'ACC':[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})

final_data

sns.barplot(x='Models', y='ACC', data=final_data)

# save the model 

rf1 = RandomForestClassifier()
rf1.fit(X_res,y_res)

import joblib

joblib.dump(rf1,'credit_card_mdl')

model=joblib.load("credit_card_mdl")

pred =model.predict([[-1.359807,-0.072781,2.536347,1.378155,-0.338321,0.462388,0.239599,0.098698,0.363787,0.090794,-0.551600,-0.617801,-0.991390,-0.311169,1.468177,-0.470401,0.207971,0.025791,0.403993,0.251412,-0.018307,0.277838,-0.110474,0.066928,0.128539,-0.189115,0.133558,-0.021053,0.244964]])

if pred==0:
    print("Normal Transaction")
else:
    print("Fraudlant Transaction")

from tkinter import *
import joblib

def show_entry_fields():
    v1=float(e1.get())
    v2=float(e2.get())
    v3=float(e3.get())
    v4=float(e4.get())
    v5=float(e5.get())
    v6=float(e6.get())

    v7=float(e7.get())
    v8=float(e8.get())
    v9=float(e9.get())
    v10=float(e10.get())
    v11=float(e11.get())
    v12=float(e12.get())

    v13=float(e13.get())
    v14=float(e14.get())
    v15=float(e15.get())
    v16=float(e16.get())
    v17=float(e17.get())
    v18=float(e18.get())


    v19=float(e19.get())
    v20=float(e20.get())
    v21=float(e21.get())
    v22=float(e22.get())
    v23=float(e23.get())
    v24=float(e24.get())


    v25=float(e25.get())
    v26=float(e26.get())
    v27=float(e27.get())
    v28=float(e28.get())
    v29=float(e29.get())


    model = joblib.load('credit_card_mdl')
    y_pred = model.predict([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,
                                v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29]])
    list1=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,
                                v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29]

    result = []
    if y_pred ==0:

        result.append("Normal Transcation")
    else:

        result.append("Fraudulent Transcation")
    print("######################################")
    print("Credit Card Fraud Detection System", result)
    print("######################################")



    Label(master, text="Final Prediction from the model - credit card fraud detection").grid(row=31)
    Label(master, text=result).grid(row=32)



master = Tk()
master.title("Credit Card Fraud Detection System")


label = Label(master, text = "Credit Card Fraud Detection System"
                          , bg = "black", fg = "white",width = 30).grid(row=0,columnspan=2)


Label(master, text="Enter value of V1").grid(row=1)
Label(master, text="Enter value of V2").grid(row=2)
Label(master, text="Enter value of V3").grid(row=3)
Label(master, text="Enter value of V4").grid(row=4)
Label(master, text="Enter value of V5").grid(row=5)
Label(master, text="Enter value of V6").grid(row=6)

Label(master, text="Enter value of V7").grid(row=7)
Label(master, text="Enter value of V8").grid(row=8)
Label(master, text="Enter value of V9").grid(row=9)
Label(master, text="Enter value of V10").grid(row=10)
Label(master, text="Enter value of V11").grid(row=11)
Label(master, text="Enter value of V12").grid(row=12)

Label(master, text="Enter value of V13").grid(row=13)
Label(master, text="Enter value of V14").grid(row=14)
Label(master, text="Enter value of V15").grid(row=15)
Label(master, text="Enter value of V16").grid(row=16)
Label(master, text="Enter value of V17").grid(row=17)
Label(master, text="Enter value of V18").grid(row=18)

Label(master, text="Enter value of V19").grid(row=19)
Label(master, text="Enter value of V20").grid(row=20)
Label(master, text="Enter value of V21").grid(row=21)
Label(master, text="Enter value of V22").grid(row=22)
Label(master, text="Enter value of V23").grid(row=23)
Label(master, text="Enter value of V24").grid(row=24)

Label(master, text="Enter value of V25").grid(row=25)
Label(master, text="Enter value of V26").grid(row=26)
Label(master, text="Enter value of V27").grid(row=27)
Label(master, text="Enter value of V28").grid(row=28)
Label(master, text="Enter value of V29").grid(row=29)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)

e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)

e13 = Entry(master)
e14 = Entry(master)
e15 = Entry(master)
e16 = Entry(master)
e17 = Entry(master)
e18= Entry(master)

e19 = Entry(master)
e20 = Entry(master)
e21 = Entry(master)
e22 = Entry(master)
e23= Entry(master)
e24 = Entry(master)


e25 = Entry(master)
e26= Entry(master)
e27 = Entry(master)
e28 = Entry(master)
e29= Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)

e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)


e13.grid(row=13, column=1)
e14.grid(row=14, column=1)
e15.grid(row=15, column=1)
e16.grid(row=16, column=1)
e17.grid(row=17, column=1)
e18.grid(row=18, column=1)


e19.grid(row=19, column=1)
e20.grid(row=20, column=1)
e21.grid(row=21, column=1)
e22.grid(row=22, column=1)
e23.grid(row=23, column=1)
e24.grid(row=24, column=1)

e25.grid(row=25, column=1)
e26.grid(row=26, column=1)
e27.grid(row=27, column=1)
e28.grid(row=28, column=1)
e29.grid(row=29, column=1)
 
Button(master, text='Predict', command=show_entry_fields).grid(row=30, column=1, sticky=W, pady=4)

mainloop( )

