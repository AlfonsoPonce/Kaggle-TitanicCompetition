import pandas as pd
from sklearn.model_selection import GridSearchCV
import sklearn.neighbors as sk
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.svm import SVC
import xgboost as xgb

def splitTrainTest(data):
    test_data = data.sample(n=round(len(data) * 0.3))
    data2 = data.drop(test_data[:].index)

    X_Train = data2.iloc[:, 1:7]
    Y_Train = data2.iloc[:, 0]

    X_Test = test_data.iloc[:, 1:7]
    Y_Test = test_data.iloc[:, 0]

    return X_Train, Y_Train, X_Test, Y_Test



def getBestModel(model_set, data):
        Models = []

        for m in model_set:
            Best_Model = hyperparameterTunning(m, data)
            Models.append(Best_Model)

        for i in range(len(Models)):
            j = i + 1
            if j < len(Models):
                comparison, best_score = comparingModel(Models[i], Models[j], data)
                if comparison == True:
                    BestOption = Models[i]
                else:
                    BestOption = Models[j]

                print("--------------------")
                print(BestOption)
                print(best_score)
                print("--------------------")

        return BestOption

def hyperparameterTunning(model, data):
    if model == "KNN":
        obj_model = sk.KNeighborsClassifier()
        print(model + " PARAMS")
        print(obj_model.get_params().keys())
        hyper_vals = {'n_neighbors' : [k+1 for k in range(20)],
                      'metric':['minkowski', 'manhattan', 'cityblock', 'cosine']}
    elif model == "BinTree":
        obj_model = tree.DecisionTreeClassifier()
        print(model + " PARAMS")
        print(obj_model.get_params().keys())
        hyper_vals = {'criterion':['gini', 'entropy'],
                      'splitter':['best', 'random'],
                      'min_samples_split': [2,3,4,5],
                      'min_impurity_decrease':[0.0, 0.05, 0.1, 0.15, 0.2]
                      }
    elif model == "KSVM":
        obj_model = SVC()
        print(model + " PARAMS")
        print(obj_model.get_params().keys())
        hyper_vals = {'C': [0.25, 0.5, 0.75],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': ['scale', 'auto']
                      }
    elif model == "XGBoost":
        return xgboostTunning(data)

    grid_lr = GridSearchCV(estimator=obj_model, param_grid=hyper_vals, scoring='accuracy',
                              cv=10, refit=True, return_train_score=True)

    X_Train, Y_Train, X_Test, Y_Test = splitTrainTest(data)
    grid_lr.fit(X_Train, Y_Train)
    preds = grid_lr.best_estimator_.predict(X_Test)
    hits = (preds == Y_Test).value_counts().loc[True]
    fails = (preds == Y_Test).value_counts().loc[False]
    accuracy = hits / (hits + fails)
    print(grid_lr.best_params_)
    print("Best Hyper Accuracy: " + str(accuracy))

    return createModel(model, grid_lr)


def comparingModel(M1, M2, data):
    Niters = 100
    best_M1 = 0
    best_M2 = 0
    best_mean_score = 0
    if type(M1) == xgb.sklearn.XGBClassifier or type(M2) == xgb.sklearn.XGBClassifier:
        data['Sex'] = pd.to_numeric(data['Sex'])

    for i in range(Niters):
        data = data.sample(frac=1)
        scores_M1 = cross_val_score(M1, data.iloc[:, 1:7], data.iloc[:, 0], cv=10)
        scores_M2 = cross_val_score(M2, data.iloc[:, 1:7], data.iloc[:, 0], cv=10)

        if scores_M1.mean() > scores_M2.mean():
            best_M1 += 1
            best_mean_score = scores_M1.mean()
        else:
            best_M2 += 1
            best_mean_score = scores_M2.mean()


    if best_M1 > best_M2:
        return True, best_mean_score
    else:
        return False, best_mean_score


def xgboostTunning(data):
    X_Train, Y_Train, X_Test, Y_Test = splitTrainTest(data)


    #ONLY FOR TITANIC CASE
    X_Train['Sex'] = pd.to_numeric(X_Train['Sex'])
    X_Test['Sex'] = pd.to_numeric(X_Test['Sex'])
    # ONLY FOR TITANIC CASE


    hyper_vals = {'max_depth': [2, 6],
                  'eta': [1],
                  'objective': ['binary:logistic'],
                  'subsample':[0.25, 0.5, 0.75, 1],
                  'n_estimators': [5,10,1000],  # number of trees, change it to 1000 for better results
                  'verbosity':[0]
                  }
    obj_model = xgb.XGBClassifier()

    grid_lr = GridSearchCV(estimator=obj_model, param_grid=hyper_vals, scoring='accuracy',
                           cv=10, refit=True, return_train_score=True)

    grid_lr.fit(X_Train, Y_Train)
    preds = grid_lr.best_estimator_.predict(X_Test)
    hits = (preds == Y_Test).value_counts().loc[True]
    fails = (preds == Y_Test).value_counts().loc[False]
    accuracy = hits / (hits + fails)
    print(grid_lr.best_params_)
    print("Best Hyper Accuracy: " + str(accuracy))

    return createModel("XGBoost", grid_lr)

def createModel(model, conf):
    if model == "KNN":
        return sk.KNeighborsClassifier(**conf.best_params_)
    elif model == "BinTree":
        return tree.DecisionTreeClassifier(**conf.best_params_)
    elif model == "KSVM":
        return SVC(**conf.best_params_)
    elif model == "XGBoost":
        return xgb.XGBClassifier(**conf.best_params_)