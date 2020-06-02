#importing dependencies
import numpy as np
import pandas as pd
from matploitlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
%matploitlib inline
#using pandas to read the database stored in the same folder
data =pd.read_csv('mnist.csv')
viewing column heads()
#extrating data from the dataset and viewing up them close
a = data.iloc[3,1:].values
#reshaping the extrated data into a reasonable size
a = a.reshape(28,28)astype('unit8')
plt.imshow(a)
#preparing the data
#separating the labels and data values
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]
#cteating train and test sizes
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size = 0.2,random_state=4)
#check data
y_train.head()
#call rf classifier
rf = RandomForestClassifier(n_estimator=100)
#fit the model
rf.fit(x_train,y_train)
#prediction on test data
pred = rf.predict(x_test)
pred
#check prediction accuracy
s = y_test.values
#calculate number of correctly predicted values
count = 0
for i in range(len(pred)):
 if pred[i] == s[i]
   count = count+1
count
# total prediction values the code was run on
len(pred)
#accuracy value
8090/8400


