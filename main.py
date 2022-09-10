import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Analysis as A
import Preprocessing as P
import ModelSelection as M
import sklearn.neighbors as sk
from collections import Counter
import csv
import random
import statsmodels.api as sm
import pylab
import matplotlib
from scipy.stats import shapiro
import scipy.stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn import tree
import xgboost as xgb
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



N = 500
PATH = '/home/alfonso/Documentos/kaggle/titanic/train.csv'
TEST_PATH = '/home/alfonso/Documentos/kaggle/titanic/test.csv'
analysis_object = A.Analysis(PATH)
preprocessing_object = P.Preprocessing(analysis_object.openFile())

test_analysis_object = A.Analysis(TEST_PATH)
test_preprocessing_object = P.Preprocessing(test_analysis_object.openFile())



data_csv = analysis_object.openFile()
test_csv = test_analysis_object.openFile()

PassengerId = test_csv["PassengerId"]
#print(csv)
#analysis_object.femaleVsMale(csv)


preprocessing_object.eraseUselessColumns()
test_preprocessing_object.eraseUselessColumns()

null_columns = preprocessing_object.detectColumnsWithNull()
test_null_columns = test_preprocessing_object.detectColumnsWithNull()

#La columna Edad tiene atributos vacíos, podría estimarse con la clase de pasajero, edad, precio del ticket parentesco, etc.
#preprocessing_object.percentageOfMissingValues(null_columns)

#preprocessing_object.MissingValuesAnalysis("Age", "Pclass")

#preprocessing_object.pieChart("Survived", 0, "Pclass")

#preprocessing_object.twoPlot("Age", "Survived")


#preprocessing_object.discreteHistogram(preprocessing_object.data["Age"])

#preprocessing_object.boxPlot('Age')

for null_col in null_columns:
    data = preprocessing_object.imputeCMC(null_col, ["Survived","Pclass","Sex"])

#preprocessing_object.boxPlot('Age')

for null_col in test_null_columns:
    test_data = test_preprocessing_object.imputeCMC(null_col, "Sex")

data = preprocessing_object.stringTransformation()
test_data = test_preprocessing_object.stringTransformation()

data = preprocessing_object.roundingAges()
#data = preprocessing_object.Normalize(['Age'],'Pow')
#data = preprocessing_object.Normalize(['Age'],'BoxCox')
#test_data = test_preprocessing_object.Normalize('MinMax')
#preprocessing_object.corr4ContinuousVsCategorical('Fare', 'Pclass', 'variance')

#exit(0)
model_set = ["KNN", "BinTree", "XGBoost", "KSVM"]

#model = M.getBestModel(model_set, data)

#print("BEST MODEL")
#print(model)
X_Train = data.iloc[:,1:7]
Y_Train = data.iloc[:,0]
X_Train['Sex'] = pd.to_numeric(X_Train['Sex'])
knn = sk.KNeighborsClassifier(n_neighbors=3)
bintree = tree.DecisionTreeClassifier(criterion= 'entropy', min_impurity_decrease= 0.0, min_samples_split= 4, splitter= 'random' )
svc = SVC(C= 0.75, gamma= 'auto', kernel= 'poly')
xgb = xgb.XGBClassifier(eta= 1, max_depth= 2, n_estimators= 5, objective= 'binary:logistic', subsample= 0.5, verbosity= 0)
#X_Train = data.iloc[:,1:7]
#Y_Train = data.iloc[:,0]


knn.fit(X_Train,Y_Train)
bintree.fit(X_Train,Y_Train)
svc.fit(X_Train,Y_Train)
X_Train['Sex'] = pd.to_numeric(X_Train['Sex'])
xgb.fit(X_Train,Y_Train)


'''
for k in range(10):
    acc_arr = []
    for i in range(N):
        val_data = data.sample(n = round(len(data)*0.3))
        data2 = data.drop(val_data[:].index)

        #classifier = sk.KNeighborsClassifier(n_neighbors=k+1)
        #classifier = tree.DecisionTreeClassifier()




        #data = data.sample(frac=1)

        #scores = cross_val_score(classifier, data.iloc[:,1:7], data.iloc[:,0], cv=10)

        X_Train = data2.iloc[:,1:7]
        Y_Train = data2.iloc[:,0]

        X_Val = val_data.iloc[:,1:7]
        Y_Val = val_data.iloc[:,0]

        #X, y = data.iloc[:, 1:7], data.iloc[:, 0]
        X_Train['Sex'] = pd.to_numeric(X_Train['Sex'])
        X_Val['Sex'] = pd.to_numeric(X_Val['Sex'])
        dtrain = xgb.DMatrix(data=X_Train, label=Y_Train)
        dval = xgb.DMatrix(data=X_Val, label=Y_Val)
        param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        num_round = 2
        bst = xgb.train(param, dtrain, num_round)
        preds = bst.predict(dval)

        #classifier.fit(X_Train,Y_Train)

        #result = classifier.predict(X_Val)

        #hits = (result == Y_Val).value_counts().loc[True]
        #fails = (result == Y_Val).value_counts().loc[False]

        #accuracy = hits / (hits + fails)

        acc_arr.append(scores.mean())
        #acc_arr.append(accuracy)
    print("K = " + str(k+1))

    print("Accuracy mean --> "+str(np.mean(acc_arr)))
    print("Accuracy std --> "+str(np.std(acc_arr)))

    #sm.qqplot(np.asarray(acc_arr), line='s')
    #pylab.show()

    stat, p = shapiro(acc_arr)
    print('stat=%.3f, p=%.3f\n' % (stat, p))
    if p > 0.05:
        print('Probably Gaussian')
    else:
        print('Probably not Gaussian')

    m, lower_lim, upper_lim = mean_confidence_interval(acc_arr)

    print("mean = " + str(m))
    print("Lower_lim = " + str(lower_lim))
    print("Upper_Lim = " + str(upper_lim))

    print("-------------------")
    print("-------------------")
    print("-------------------")













'''

#result = classifier.predict(test_data)
knn_result = knn.predict(test_data)
bintree_result = bintree.predict(test_data)
svc_result = svc.predict(test_data)
test_data['Sex'] = pd.to_numeric(test_data['Sex'])
xgb_result = xgb.predict(test_data)

#result = xgb_result.copy() 
result = np.zeros(len(knn_result), dtype=int)
for i in range(len(xgb_result)):
    votes = [bintree_result[i], xgb_result[i], knn_result[i], svc_result[i]]
    if votes[0] == votes[1] == votes[2] == votes[3]:
        result[i] = votes[0]
    '''
    is_zero = 0
    is_one = 0
    for v in votes:
        if v == 1:
            is_one += 1
        else:
            is_zero += 1

    if is_one == is_zero:
        result[i] = xgb_result[i]
    else:
        if is_one > is_zero:
            result[i] = 1
        else:
            result[i] = 0
    '''
header = ["PassengerId", "Survived"]




with open('submission.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)

    writer.writerow(header)

    # write a row to the csv file
    for i in range(len(PassengerId)):
        writer.writerow([PassengerId[i], result[i]])

    f.close()


