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
    return(roc_auc_score(y_test, y_pred, average=average))


class BestEstimator(object):

    def __init__(self, Data, Target):

        self.Data = Data.copy()
        self.Target = Target.copy()
        self.dim_ = Data.shape

        self.AUC = make_scorer(multiclass_roc_auc_score)

    def best_estim(self,
                 type_esti='classifier',  # Type of estimator : classifier or regressor
                 params=False,  # Allow to use a custom hyperparametres dict for GridSearCV
                 ID='ID',  # ID feature of the DataFrame used
                 target_ID=True,  # If Target feature have an ID
                 cv=3,  # Numbers of folds for the first estimators check
                 grid=False,  # if True, use a GridSearchCV with best estimator found
                 cv_grid=3,  # Number of folds for the GridSearchCV
                 n=10000,  # Number of observations used for the first check
                 n_grid=10000,  # Number of observations used for the GridSearchCV
                 value=0,  # Value for fill NaN
                 view_nan=False,  # if True check the NaN Data
                 scoring='mae'):  # Type of scorer

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
            print('\n NaN data filled by {} \n'.format(value))
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
                le = LabelEncoder()
                le.fit(list(self.Target[i]))
                self.Target[i] = le.transform(list(self.Target[i]))

        X_tr, X_te, Y_tr, Y_te = train_test_split(self.Data, self.Target, random_state=0, test_size=1 / 3)

        print('\n Searching for the best regressor on {} data using {} loss... \n'.format(n, scoring))

        if type_esti == 'classifier':

            # print('\n Searching for the best classifier on {} data... \n'.format(n))

            clfs = {}
            clfs['Bagging'] = {'clf': BaggingClassifier(), 'name': 'Bagging'}
            clfs['Gradient Boosting'] = {'clf': GradientBoostingClassifier(), 'name': 'Gradient Boosting'}
            clfs['XGBoost'] = {'clf': XGBClassifier(), 'name': 'XGBoost'}
            clfs['Random Forest'] = {'clf': RandomForestClassifier(n_estimators=100, n_jobs=-1),
                                     'name': 'Random Forest'}
            clfs['Decision Tree'] = {'clf': DecisionTreeClassifier(), 'name': 'Decision Tree'}
            clfs['KNN'] = {'clf': KNeighborsClassifier(n_jobs=-1), 'name': 'KNN'}
            # clfs['NN'] = {'clf': MLPClassifier(), 'name': 'MLPClassifier'
            # clfs['LR'] = {'clf': LogisticClassifier(), 'name': 'LR'}
            clfs['SVM'] = {'clf': SVC(gamma='auto'), 'name': 'SVM'}

            for item in clfs:
                Score = cross_val_score(clfs[item]['clf'], np.asarray(X_tr[0:n]), np.ravel(Y_tr[0:n]),
                                        cv=cv, scoring=self.AUC)
                Score_mean = Score.mean()
                STD2 = Score.std() * 2

                clfs[item]['score'] = Score  # roc_auc
                clfs[item]['mean'] = Score_mean
                clfs[item]['std2'] = STD2

                print("{} \n".format(item + ": %0.4f (+/- %0.4f)" % (clfs[item]['score'].mean(),
                                                                     clfs[item]['score'].std() * 2)))

            Best_clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['name']


        elif type_esti == 'regressor':

            clfs = {}
            clfs['Bagging'] = {'clf': BaggingRegressor(), 'name': 'Bagging'}
            clfs['Gradient Boosting'] = {'clf': GradientBoostingRegressor(), 'name': 'Gradient Boosting'}
            # clfs['XGBoost'] = {'clf': XGBRegressor(), 'name': 'XGBoost'}
            clfs['Random Forest'] = {'clf': RandomForestRegressor(n_estimators=100, n_jobs=-1),
                                     'name': 'Random Forest'}
            clfs['Decision Tree'] = {'clf': DecisionTreeRegressor(), 'name': 'Decision Tree'}
            clfs['KNN'] = {'clf': KNeighborsRegressor(n_jobs=-1), 'name': 'KNN'}
            # clfs['NN'] = {'clf': MLPClassifier(), 'name': 'MLPClassifier'
            # clfs['LR'] = {'clf': LogisticClassifier(), 'name': 'LR'}
            clfs['SVM'] = {'clf': SVR(gamma='auto'), 'name': 'SVM'}

            for item in clfs:
                Score = cross_val_score(clfs[item]['clf'], np.asarray(X_tr[0:n]), np.ravel(Y_tr[0:n]),
                                        cv=cv, scoring=scoring)
                Score_mean = Score.mean()
                STD2 = Score.std() * 2

                clfs[item]['score'] = Score  # roc_auc
                clfs[item]['mean'] = Score_mean
                clfs[item]['std2'] = STD2

                print("{} \n".format(item + ": %0.4f (+/- %0.4f)" % (clfs[item]['score'].mean(),
                                                                     clfs[item]['score'].std() * 2)))

            Best_clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['name']

        if grid:
            # print('grid = True')

            if params == False:
                # print('params = False')

                # print(Best_clf)

                if Best_clf == 'Gradient Boosting':
                    #   print('Best_clf = gb')

                    params = {'n_estimators': [100, 300, 600],
                              'max_depth': [5, 10, None],
                              'learning_rate': [.001, .01, .1]}


                elif Best_clf == 'Random Forest':
                    #  print('Best_clf = dt ou rf')

                    params = {'n_estimators': [10, 100, 300],
                              'max_depth': [5, 10, None],
                              'criterion': ['gini', 'entropy']}

                elif Best_clf == 'Decision Tree':
                    # print('best_clf = dt')

                    params = {'max_depth': [5, 10, 50, None],
                              'criterion': ['gini', 'entropy']}

                elif Best_clf == 'XGBoost':
                    # print('Best_clf = xgb')

                    params = {'eta': [.01, .1, .3],
                              'max_depth': [5, 10, None],
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

            print('\n Searching for best hyperparametres of {} on {} data among : \n'.format(Best_clf, n_grid))
            print('{} \n'.format(params))
            # print('Starting GridSearchCV using {} Classifier with {} folds \n'.format(Best_clf, cv_grid))

            if scoring == 'AUC':
                scoring = self.AUC

            clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['clf']
            gr = GridSearchCV(clf, param_grid=params, cv=cv_grid, scoring=scoring, n_jobs=-1,
                              verbose=1, refit=True, iid=True)  # ;

            gr.fit(X_tr[0:n_grid], np.ravel(Y_tr[0:n_grid]))

            # print(' Best score :', gr.best_score_,   '\n Using these parametres :', gr.best_params_)

            #####
            print('\n Finally, best estimator is : {} Classifier'.format(Best_clf), '\n Using these parametres :',
                  gr.best_params_,
                  '\n With this score : {}'.format(gr.best_score_))
            #####
            return (gr)

    def grid(self, clf, params, cv=3, n=100000):

        X_tr, X_te, Y_tr, Y_te = train_test_split(self.Data, self.Target, random_state=0, test_size=1 / 3)

        gr = GridSearchCV(clf, param_grid=params, cv=cv, scoring=self.AUC, n_jobs=-1,
                          verbose=1, refit=True, iid=True);

        gr.fit(X_tr[0:n], np.ravel(Y_tr[0:n]))

        # print(' Best score :', gr.best_score_,   '\n Using this parametres :', gr.best_params_, '\n With :', clf)
        print(' Best score on Train:', gr.best_score_, '\n Using this parametres :', gr.best_params_,
              '\n With : \n {} '.format(clf))
        return gr

    def feature_eng(self, Test, value=0, ID='ID'):

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

    def pred(self, Test, gr, prob=False, same=True, ID='ID', value=0):  #

        # Test.drop([ID], axis = 1, inplace = True)
        Pred = pd.DataFrame()

        if same == False:

            Test.drop([ID], axis=1, inplace=True)

            if type(value) == int:
                Test.fillna(value, inplace=True)

            elif value == 'bfill':
                Test.fillna('bfill', inplace=True)

            elif value == 'ffill':
                Test.fillna('ffill', inplace=True)

            for i in Test.columns:
                if Test[i].dtype == float:
                    Test[i] = Test[i].astype('int')

                elif Test[i].dtype == object:
                    encoder = LabelEncoder()
                    encoder.fit(list(Test[i]))
                    Test[i] = encoder.transform(list(Test[i]))

        if prob == False:
            # Pred[ID] = Test[ID]
            Pred['Target'] = gr.predict(Test)
            return (Pred)

        else:
            return (gr.predict_proba(Test))

    # else :
    #    return(gr.predict_proba(self.feature_eng(Data, value , ID)))






