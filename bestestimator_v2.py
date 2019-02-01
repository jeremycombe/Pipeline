import pandas as pd
import numpy as np
import operator

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor, \
    RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return (roc_auc_score(y_test, y_pred, average=average))


class BestEstimator(object):

    def __init__(self, 
                 type_esti = 'classifier', 
                 cv = 3, 
                 grid = True, 
                 hard_grid = False,
                 cv_grid = 3):
        
        self.type_esti = type_esti
        #self.params = params
        self.cv = cv
        self.grid = grid
        self.hard_grid = hard_grid
        self.cv_grid = cv_grid
        
        self.AUC = make_scorer(multiclass_roc_auc_score)
        
        self.Decision_Function = None
        self.gr = None
        self.estim = None
        self.Target = None
        self.Data = None
        self.le = None
        self.cat = None
        

    def fit(self, data, target,
            ID = 'ID',
            target_ID = True,
            n = 1000,
            n_grid = 1000,
            value_nan = 0,
            view_nan = True,
           params = False,
           value = 0,
           scoring = 'AUC'):
        
        loss = scoring 
        self.Data = data.copy()
        self.Target = target.copy()

        self.Data.drop([ID], axis=1, inplace=True)
        if target_ID:
            self.Target.drop([ID], axis=1, inplace=True)

        if view_nan:
            print("Missing Values :\n")

            total = self.Data.isnull().sum().sort_values(ascending=False)
            percent = (self.Data.isnull().sum() / self.Data.isnull().count()).sort_values(ascending=False) * 100
            missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
            print("{} \n".format(missing_data[(percent > 0)]))

        if type(value) == int:
            self.Data.fillna(value, inplace=True)
            # self.Test.fillna(value, inplace = True)
            # self.Missing_values()

        elif value == 'bfill':
            self.Data.fillna('bfill', inplace=True)
            # self.Test.fillna('bfill', inplace = True)
            # self.Missing_values()

        elif value == 'ffill':
            self.Data.fillna('ffill', inplace=True)
            # self.Test.fillna('ffill', inplace = True)
            # self.Missing_values()

        if self.Data.isnull().any().any() == False:
            print('NaN data filled by {} \n'.format(value))
        else:
            print('Fail to fill NaN data')

        for i in self.Data.columns:  ###########

            if self.Data[i].dtype == object:
                encoder = LabelEncoder()
                encoder.fit(list(self.Data[i]))
                self.Data[i] = encoder.transform(list(self.Data[i]))

            if self.Data[i].dtype == float:
                self.Data[i] = self.Data[i].astype('int')

        for i in self.Target.columns:
            if self.Target[i].dtype == object:
                self.cat = True
                self.le = LabelEncoder()
                self.le.fit(list(self.Target[i]))
                self.Target[i] = self.le.transform(list(self.Target[i]))

        X_tr, X_te, Y_tr, Y_te = train_test_split(self.Data, self.Target, random_state=0, test_size=1 / 3)

        print('Searching for the best regressor on {} data using {} loss... \n'.format(n, scoring))

        if self.type_esti == 'classifier':

            # print('\n Searching for the best classifier on {} data... \n'.format(n))

            clfs = {}
            clfs['Bagging'] = {'clf': BaggingClassifier(), 'name': 'Bagging'}
            clfs['Gradient Boosting'] = {'clf': GradientBoostingClassifier(), 'name': 'Gradient Boosting'}
            clfs['XGBoost'] = {'clf': XGBClassifier(), 'name': 'XGBoost'}
            clfs['Random Forest'] = {'clf': RandomForestClassifier(n_estimators=100, n_jobs=-1),
                                     'name': 'Random Forest'}
            clfs['Decision Tree'] = {'clf': DecisionTreeClassifier(), 'name': 'Decision Tree'}
            clfs['Extra Tree'] = {'clf': ExtraTreesClassifier(n_jobs=-1), 'name': 'Extra Tree'}

            clfs['KNN'] = {'clf': KNeighborsClassifier(n_jobs=-1), 'name': 'KNN'}
            # clfs['NN'] = {'clf': MLPClassifier(), 'name': 'MLPClassifier'
            # clfs['LR'] = {'clf': LogisticClassifier(), 'name': 'LR'}
            clfs['SVM'] = {'clf': SVC(gamma='auto'), 'name': 'SVM'}

            
            if scoring == 'AUC' and np.unique(self.Target).shape[0] > 2:
                scoring = self.AUC
                score = 'AUC'
            else :
                score = 'AUC'
                scoring = 'roc_auc'
            
            for item in clfs:
                
                Score = cross_val_score(clfs[item]['clf'], np.asarray(X_tr[0:n]), np.ravel(Y_tr[0:n]),
                                        cv=self.cv, scoring=scoring)
               
                Score_mean = Score.mean()
                STD2 = Score.std() * 2

                clfs[item]['score'] = Score  # roc_auc
                clfs[item]['mean'] = Score_mean
                clfs[item]['std2'] = STD2

                print("\n {}".format(item + ": %0.4f (+/- %0.4f)" % (clfs[item]['score'].mean(),
                                                                     clfs[item]['score'].std() * 2)))

            Best_clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['name']


        elif self.type_esti == 'regressor':

            clfs = {}
            clfs['Bagging'] = {'clf': BaggingRegressor(), 'name': 'Bagging'}
            clfs['Gradient Boosting'] = {'clf': GradientBoostingRegressor(), 'name': 'Gradient Boosting'}
            clfs['XGBoost'] = {'clf': XGBRegressor(), 'name': 'XGBoost'}
            clfs['Random Forest'] = {'clf': RandomForestRegressor(n_estimators=100, n_jobs=-1),
                                     'name': 'Random Forest'}
            clfs['Decision Tree'] = {'clf': DecisionTreeRegressor(), 'name': 'Decision Tree'}
            clfs['Extra Tree'] = {'clf': ExtraTreesRegressor(n_jobs=-1), 'name': 'Extra Tree'}
            clfs['KNN'] = {'clf': KNeighborsRegressor(n_jobs=-1), 'name': 'KNN'}
            # clfs['NN'] = {'clf': MLPClassifier(), 'name': 'MLPClassifier'
            # clfs['LR'] = {'clf': LogisticClassifier(), 'name': 'LR'}
            clfs['SVM'] = {'clf': SVR(gamma='auto'), 'name': 'SVM'}

            for item in clfs:
                # print(Y_tr[0:30])
                Score = cross_val_score(clfs[item]['clf'], np.asarray(X_tr[0:n]), np.array(np.ravel(Y_tr[0:n])),
                                        ########""
                                        cv=self.cv, scoring=scoring)
                Score_mean = Score.mean()
                STD2 = Score.std() * 2

                clfs[item]['score'] = Score  # roc_auc
                clfs[item]['mean'] = Score_mean
                clfs[item]['std2'] = STD2

                print("\n {}".format(item + ": %0.4f (+/- %0.4f)" % (clfs[item]['score'].mean(),
                                                                     clfs[item]['score'].std() * 2)))

            Best_clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['name']

        if self.grid:
            # print('grid = True')

            if params == False:

                if self.hard_grid == False:

                    if Best_clf == 'Extra Tree':

                        if self.type_esti == 'regressor':

                            params = {'n_estimators': [100, 300, 600],
                                      'criterion': ['mse', 'mae'],
                                      'max_depth': [None, 5, 10]}

                        else:

                            params = {'n_estimators': [100, 300, 600],
                                      'criterion': ['gini', 'entropy'],
                                      'max_depth': [None, 5, 10]}

                    if Best_clf == 'Gradient Boosting':

                        if self.type_esti == 'regressor':

                            params = {'n_estimators': [100, 300, 600],
                                      'max_depth': [5, 10, None],
                                      'learning_rate': [.001, .01, .1],
                                      'loss': ['ls', 'lad']}
                        else:

                            params = {'n_estimators': [100, 300, 600],
                                      'max_depth': [5, 10, None],
                                      'learning_rate': [.001, .01, .1],
                                      'loss': ['deviance', 'exponential']}


                    elif Best_clf == 'Random Forest':
                        #  print('Best_clf = dt ou rf')

                        if self.type_esti == 'regressor':

                            params = {'n_estimators': [10, 100, 300],
                                      'max_depth': [5, 10, None],
                                      'criterion': ['mse', 'mae']}

                        else:

                            params = {'n_estimators': [10, 100, 300],
                                      'max_depth': [5, 10, None],
                                      'criterion': ['gini', 'entropy']}

                    elif Best_clf == 'Decision Tree':

                        if self.type_esti == 'regressor':

                            params = {'max_depth': [5, 10, 50, None],
                                      'criterion': ['mse', 'friedman_mse', 'mae']}

                        else:

                            params = {'max_depth': [5, 10, 50, None],
                                      'criterion': ['gini', 'entropy']}


                    elif Best_clf == 'XGBoost':
                        # print('Best_clf = xgb')

                        params = {'eta': [.01, .1, .3],
                                  'max_depth': [5, 10, 15],
                                  'gamma': [0, .1, .01]}

                    elif Best_clf == 'Bagging':
                        # print('best_clf = bag)')

                        params = {'n_estimators': [100, 300, 600]}

                    elif Best_clf == 'KNN':

                        params = {'n_neighbors': [2, 5, 10, 30, 40],
                                  'p': [1, 2]}

                    elif Best_clf == 'SVM':

                        params = {'C': {1, .5, .1, 5},
                                  'tol': [.01, .001, .1, .0001]}



                else:

                    if Best_clf == 'Extra Tree':

                        if self.type_esti == 'regressor':

                            params = {'n_estimators': [10, 100, 300, 600, 1000, 1200],
                                      'criterion': ['mae', 'mse'],
                                      'max_depth': [None, 5, 10, 15, 20, 25]}

                        else:

                            params = {'n_estimators': [10, 100, 300, 600, 1000, 1200],
                                      'criterion': ['gini', 'entropy'],
                                      'max_depth': [None, 5, 10, 15, 20, 25]}

                    if Best_clf == 'Gradient Boosting':

                        if self.type_esti == 'regressor':

                            params = {'n_estimators': [100, 300, 600, 1000, 1200],
                                      'max_depth': [5, 10, 15, 25, None],
                                      'learning_rate': [.001, .01, .1],
                                      'loss': ['ls', 'lad', 'huber', 'quantile'],
                                      'criterion': ['mse', 'friedman_mse']}
                        else:

                            params = {'n_estimators': [100, 300, 600, 1000, 1200],
                                      'max_depth': [5, 10, 15, 25, None],
                                      'learning_rate': [.001, .01, .1],
                                      'loss': ['deviance', 'exponential'],
                                      'criterion': ['mse', 'friedman_mse']}


                    elif Best_clf == 'Random Forest':
                        #  print('Best_clf = dt ou rf')

                        if self.type_esti == 'regressor':

                            params = {'n_estimators': [10, 100, 300, 600, 1000, 1200],
                                      'max_depth': [5, 10, 15, 20, 25, None],
                                      'criterion': ['mse', 'mae']}

                        else:

                            params = {'n_estimators': [10, 100, 300, 600, 1000, 1200],
                                      'max_depth': [5, 10, 15, 20, 25],
                                      'criterion': ['gini', 'entropy']}

                    elif Best_clf == 'Decision Tree':

                        if params == 'regressor':

                            params = {'max_depth': [5, 10, 50, 100, None],
                                      'criterion': ['mse', 'friedman_mse', 'mae'],
                                      'splitter': ['best', 'random']}

                        else:

                            params = {'max_depth': [5, 10, 50, 100, None],
                                      'criterion': ['gini', 'entropy'],
                                      'splitter': ['best', 'random']}


                    elif Best_clf == 'XGBoost':
                        # print('Best_clf = xgb')

                        params = {'eta': [0.001, .01, .1, .3, 1],
                                  'max_depth': [5, 10, 15, 20, 25],
                                  'gamma': [0, .1, .01, .001]}

                    elif Best_clf == 'Bagging':
                        # print('best_clf = bag)')

                        params = {'n_estimators': [100, 300, 600, 1000, 1200, 1500]}

                    elif Best_clf == 'KNN':

                        params = {'n_neighbors': [2, 5, 10, 30, 40, 70, 100],
                                  'p': [1, 2, 3]}

                    elif Best_clf == 'SVM':

                        params = {'C': {1, .5, .1, 5, .01, .001},
                                  'tol': [.01, .001, .1, .0001, 1],
                                  'kernel': ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']}

            if self.hard_grid:
                print('\n Searching for the best hyperparametres of {} using hard_grid on {} data among : \n'.format(
                    Best_clf, n_grid))

            else:
                print('\n Searching for the best hyperparametres of {} on {} data among : \n'.format(Best_clf, n_grid))
            print('{} \n'.format(params))
            # print('Starting GridSearchCV using {} Classifier with {} folds \n'.format(Best_clf, cv_grid))

            
        
            clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['clf']

            if loss == 'AUC' and np.unique(self.Target).shape[0] > 2:
                
                
                self.gr = GridSearchCV(clf, param_grid=params, cv=self.cv_grid, scoring=scoring, 
                                verbose=1, refit=True, iid=True)
            else :
                self.gr = GridSearchCV(clf, param_grid=params, cv=self.cv_grid, scoring=scoring, 
                                    verbose=1, refit=True, iid=True, n_jobs = -1)

            self.gr.fit(X_tr[0:n_grid], np.ravel(Y_tr[0:n_grid]))
            


            print('\n Finally, the best estimator is : {} {}'.format(Best_clf, self.type_esti))

            print('\n Using these hyperparametres : {}'.format(self.gr.best_params_))

            print('\n With this {} score : {}'.format(loss, self.gr.best_score_))
            
            self.Decision_Function = self.gr.best_estimator_
            
            
        else:
            print('\n Best {} : {}'.format(self.type_esti, Best_clf))
            
            
            
    def ReFit(self, Train, Target, ID = 'ID', target_ID = 'ID', value = 0):
        
        train = Train.copy()
        target = Target.copy()
        
        train = self.Transform(train, value = value, ID = ID)
        target = self.Transform(target, value = value, ID = ID)
        
        self.estim = self.Decision_Function.fit(train, target)
        
        return(self.estim)

    
    
    def Transform(self, Data, value=0, ID = 'ID'):
        
        Test = Data.copy()
        
        if ID != None:

            Test.drop([ID], axis=1, inplace=True)

        if type(value) == int:
            Test.fillna(value, inplace=True)

        elif value == 'bfill':
            Test.fillna('bfill', inplace=True)

        elif value == 'ffill':
            Test.fillna('ffill', inplace=True)

        for i in Test.columns:  ###########
            if Test[i].dtype == float:
                Test[i] = Test[i].astype('int')

            elif Test[i].dtype == object:
                encoder = LabelEncoder()
                encoder.fit(list(Test[i]))
                Test[i] = encoder.transform(list(Test[i]))
        return(Test)
    
    
    
    def pred_grid(self, Test, ID = 'ID', value = 0, n = 100, prob = False):

        if ID == None:
            test = self.Transform(Test, ID = None, value = 0).copy()
        else :
            test = self.Transform(Test, ID = ID, value = 0).copy()
            
        if prob == False:
            
            pred = pd.DataFrame()

            if self.cat :
                predict = self.le.inverse_transform(self.gr.predict(test))
            else:
                predict = self.gr.predict(test)

            if ID == None:
                pred['Target'] = predict
            else :
                pred[ID] = Test[ID]
                pred['Target'] = predict

        else:
            predict = self.gr.predict_proba(test)

            if ID == None:
                pred['Target'] = predict
            else :
                pred[ID] = Test[ID]
                pred['Target'] = predict
            
        return(pred)
    
    
    
    
    def pred(self, Test, ID=None, value = 0, target_ID = None, n = 100, prob = False):
        
        if ID == None:
            test = self.Transform(Test, ID = None, value = 0).copy()
        else :
            test = self.Transform(Test, ID = ID, value = 0).copy()
            
    
        if self.estim == None:
            self.estim = self.ReFit(self.Data[0:n], self.Target[0:n], ID = None, target_ID = None, value = 0)
            if prob == False:
                pred = pd.DataFrame()
                
                if self.cat :
                    predict = self.le.inverse_transform(self.estim.predict(test))
                else:
                    predict = self.estim.predict(test)
                
                if ID == None:
                    pred['Target'] = predict
                else :
                    pred[ID] = Test[ID]
                    pred['Target'] = predict
                
            else:
                predict = self.estim.predict_proba(test)
                
                if ID == None:
                    pred['Target'] = predict
                else :
                    pred[ID] = Test[ID]
                    pred['Target'] = predict
            
        else:
            pred = pd.DataFrame()
            if prob == False:
                
                if self.cat :
                    predict = self.le.inverse_transform(self.estim.predict(test))
                else:
                    predict = self.estim.predict(test)
                
                if ID == None:
                    pred['Target'] = predict
                else :
                    pred[ID] = Test[ID]
                    pred['Target'] = predict
            else:
                predict = self.estim.predict_proba(test)
                
                if ID == None:
                    pred['Target'] = predict
                else :
                    pred[ID] = Test[ID]
                    pred['Target'] = predict
            
        return(pred)

            
            

    
        

    def grid(self, clf, params, cv=3, n=100000):

        X_tr, X_te, Y_tr, Y_te = train_test_split(self.Data, self.Target, random_state=0, test_size=1 / 3)

        gr = GridSearchCV(clf, param_grid=params, cv=cv, scoring=self.AUC, n_jobs=-1,
                          verbose=1, refit=True, iid=True);

        gr.fit(X_tr[0:n], np.ravel(Y_tr[0:n]))

        # print(' Best score :', gr.best_score_,   '\n Using this parametres :', gr.best_params_, '\n With :', clf)
        print(' Best score on Train:', gr.best_score_, '\n Using this parametres :', gr.best_params_,
              '\n With : \n {} '.format(clf))
        return gr





