import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics

df = pd.read_csv("Weather_Data.csv")

df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)
df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

x_train, x_test, y_train, y_test = train_test_split(features,Y,test_size = 0.2,random_state = 10)
LinearReg = LinearRegression()
LinearReg.fit(x_train,y_train)
predictions = LinearReg.predict(x_test)


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
LinearRegression_MAE = mean_absolute_error(predictions,y_test)
LinearRegression_MSE = mean_squared_error(predictions,y_test)
LinearRegression_R2 = r2_score(predictions,y_test)

Report = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'R-squared (R2)'],
    'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
})

#KNN
x_train_norm = preprocessing.StandardScaler().fit(x_train).transform(x_train.astype(float))
KNN = KNeighborsClassifier(n_neighbors=4).fit(x_train_norm,y_train)
#predict
x_test_norm = preprocessing.StandardScaler().fit(x_test).transform(x_test.astype(float))
KNN_predictions = KNN.predict(x_test_norm)

KNN_Accuracy_Score = accuracy_score(y_test,KNN_predictions)
KNN_JaccardIndex = jaccard_score(y_test,KNN_predictions)
KNN_F1_Score = f1_score(y_test,KNN_predictions)

new_report = pd.DataFrame({
    'Metric': ['Accuracy Score', 'Jaccard Index', 'F1 Score'],
    'Value': [KNN_Accuracy_Score, KNN_JaccardIndex, KNN_F1_Score]
})
#print(new_report)
#Decision Tree
from sklearn.utils import compute_sample_weight
w_train = compute_sample_weight('balanced', y_train)
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Tree.fit(x_train,y_train,sample_weight=w_train)
Tree_predictions = Tree.predict(x_test)
#print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, Tree_predictions))
Tree_accuracy = metrics.accuracy_score(y_test, Tree_predictions)
Tree_jaccard = jaccard_score(y_test, Tree_predictions)
Tree_f1 = f1_score(y_test, Tree_predictions)
#print(Tree_accuracy, Tree_jaccard, Tree_f1)
#Logistic Regression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
yhats = LR.predict(x_test)
yhat_prob = LR.predict_proba(x_test)
LR_jaccard = jaccard_score(y_test, yhats)
LR_f1 = f1_score(y_test, yhats)
LR_logloss = log_loss(y_test, yhat_prob)
LR_accuracy = accuracy_score(y_test, yhats)
#print(LR_accuracy, LR_jaccard, LR_f1, LR_logloss)

#SVM
SVM = svm.SVC(kernel='rbf')
SVM.fit(x_train, y_train)
SVM_predictions = SVM.predict(x_test)
SVM_accuracy = accuracy_score(y_test, SVM_predictions)
SVM_jaccard = jaccard_score(y_test, SVM_predictions,pos_label=0.0)
SVM_f1 = f1_score(y_test, SVM_predictions,average='weighted')
print(SVM_accuracy, SVM_jaccard, SVM_f1)

Report = pd.DataFrame({
    'Metric': ['Accuracy Score', 'Jaccard Index', 'F1 Score'],
    'Value': [SVM_accuracy, SVM_jaccard, SVM_f1]
})
print(Report)
