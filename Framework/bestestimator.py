import pandas as pd
import numpy as np
import operator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
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
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFpr, SelectFwe,\
    f_regression, mutual_info_regression, SelectFdr, SelectPercentile, SelectFwe
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


class BestEstimator(object):

    def __init__(self,
                 type_esti='Classifier'):

        """
        :param type_esti: Classifier or Regressor
        """

        self.type_esti = type_esti

        self.Decision_Function = None
        self.gr = None
        self.estim = None
        self.Target = None
        self.Data = None
        self.le = None
        self.lab = None
        self.lab_num = None
        self.best_score = None
        self.neg = ['neg_mean_absolute_error','neg_mean_squared_error','neg_mean_squared_log_error',
                    'neg_median_absolute_error']
        self.neg_result = None
        self.X_tr = None
        self.Y_tr = None
        self.X_te = None
        self.Y_te = None

    def fit(self, data, target,
            ID='ID',
            target_ID=True,
            n=1000,
            n_grid=1000,
            view_nan=True,
            value=0,
            scoring='roc_auc',
            cv=3,
            cv_grid = 3,
            grid = True,
            hard_grid = False):

        """
        Fit all Machine Learning algorithms on a train and target dataset, afterward
        search for the best hyperparametres of the best algorithm and return them with the loss score.

        :param data: training dataset
        :param target: target dataset
        :param ID: the ID column of the train dataset
        :param target_ID: if True, drop the ID column of the target dataset
        :param n: size of the sample for the first algorithms check
        :param n_grid: size of the sample for the GridSearchCV
        :param view_nan: if True, display some statistics on missing values
        :param value: the value for fill missing values
        :param scoring: loss function to check the estimator performance
        :param cv: fold number for the cross validation first check
        :param grid: if True, do a GridSearchCV
        :param hard_grid: if True, do a GridSearchCV with a large set of hyperparamatres
        :param cv_grid: fold number for the GridSearchCV
        """

        loss = scoring
        self.Data = data.copy()
        self.Target = target.copy()

        if self.type_esti == 'Regressor' and loss in self.neg:
            self.neg_result = True

        self.Data.drop([ID], axis=1, inplace=True)
        if target_ID:
            self.Target.drop([ID], axis=1, inplace=True)

        if view_nan:


            if self.Data.isnull().values.any():

                print("Missing Values :\n")

                total = self.Data.isnull().sum().sort_values(ascending=False)
                percent = (self.Data.isnull().sum() / self.Data.isnull().count()).sort_values(ascending=False) * 100
                missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
                print("{} \n".format(missing_data[(percent > 0)]))

                if type(value) == int:
                    self.Data.fillna(value, inplace=True)

                elif value == 'bfill':
                    self.Data.fillna('bfill', inplace=True)

                elif value == 'ffill':
                    self.Data.fillna('ffill', inplace=True)

                if self.Data.isnull().any().any() == False:
                    print('Missing values filled by {} \n'.format(value))
                else:

                    print('Fail to fill missing values')

            else :
                print('No missing values \n')

        for i in self.Data.columns:  ###########

            if self.Data[i].dtype == object:
                encoder = LabelEncoder()
                encoder.fit(list(self.Data[i]))
                self.Data[i] = encoder.transform(list(self.Data[i]))

            if self.Data[i].dtype == float:
                self.Data[i] = self.Data[i].astype('int')


        for i in self.Target.columns:
            if self.Target[i].dtype == object:
                #self.cat = True
                self.le = LabelEncoder()
                self.le.fit(list(self.Target[i]))
                self.Target[i] = self.le.transform(list(self.Target[i]))
            else:
                self.lab_num = True

        self.X_tr, self.X_te, self.Y_tr, self.Y_te = train_test_split(self.Data, self.Target, random_state=0, test_size=1 / 3)

        print('Searching for the best {} on {} datas using {} loss... \n'.format(self.type_esti,n, scoring))

        if self.type_esti == 'Classifier':

            clfs = {}
            clfs['Bagging'] = {'clf': BaggingClassifier(random_state = 0), 'name': 'Bagging'}
            clfs['Gradient Boosting'] = {'clf': GradientBoostingClassifier(random_state = 0), 'name': 'Gradient Boosting'}
            clfs['XGBoost'] = {'clf': XGBClassifier(random_state = 0), 'name': 'XGBoost'}
            clfs['Random Forest'] = {'clf': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state = 0),
                                     'name': 'Random Forest'}
            clfs['Decision Tree'] = {'clf': DecisionTreeClassifier(), 'name': 'Decision Tree'}
            clfs['Extra Tree'] = {'clf': ExtraTreesClassifier(n_jobs=-1, n_estimators = 100, random_state = 0), 'name': 'Extra Tree'}
            clfs['KNN'] = {'clf': KNeighborsClassifier(n_jobs=-1), 'name': 'KNN'}
            clfs['SVM'] = {'clf': SVC(gamma='auto'), 'name': 'SVM'}



            for item in clfs:
                Score = cross_val_score(clfs[item]['clf'], np.asarray(self.X_tr[0:n]), np.ravel(self.Y_tr[0:n]),
                                        cv=cv, scoring=scoring)

                Score_mean = Score.mean()
                STD2 = Score.std() * 2

                clfs[item]['score'] = Score
                clfs[item]['mean'] = Score_mean
                clfs[item]['std2'] = STD2

                print("\n {}".format(item + ": %0.6f (+/- %.3e)" % (clfs[item]['score'].mean(),
                                                                    clfs[item]['score'].std() ** 2)))

            Best_clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['name']

        elif self.type_esti == 'Regressor':

            clfs = {}
            clfs['Bagging'] = {'clf': BaggingRegressor(random_state = 0), 'name': 'Bagging'}
            clfs['Gradient Boosting'] = {'clf': GradientBoostingRegressor(), 'name': 'Gradient Boosting'}
            clfs['XGBoost'] = {'clf': XGBRegressor(random_state = 0), 'name': 'XGBoost'}
            clfs['Random Forest'] = {'clf': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state = 0),
                                     'name': 'Random Forest'}
            clfs['Decision Tree'] = {'clf': DecisionTreeRegressor(), 'name': 'Decision Tree'}
            clfs['Extra Tree'] = {'clf': ExtraTreesRegressor(n_jobs=-1, n_estimators = 100, random_state = 0), 'name': 'Extra Tree'}
            clfs['KNN'] = {'clf': KNeighborsRegressor(n_jobs=-1), 'name': 'KNN'}
            clfs['SVM'] = {'clf': SVR(gamma='auto'), 'name': 'SVM'}

            for item in clfs:

                Score = cross_val_score(clfs[item]['clf'], np.asarray(self.X_tr[0:n]), np.array(np.ravel(self.Y_tr[0:n])),
                                        cv=cv, scoring=scoring)
                Score_mean = Score.mean()
                STD2 = Score.std() * 2

                if self.neg_result:
                    clfs[item]['score'] = -Score
                else:
                    clfs[item]['score'] = Score
                clfs[item]['mean'] = Score_mean
                clfs[item]['std2'] = STD2

                print("\n {}".format(item + ": %0.6f (+/- %.3e)" % (clfs[item]['score'].mean(),
                                                                     clfs[item]['score'].std() ** 2)))

            if self.neg_result:
                Best_clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['name']
            else:
                Best_clf = clfs[min(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['name']
                print(Best_clf)

        if grid:
            if hard_grid == False:

                if Best_clf == 'Extra Tree':

                    if self.type_esti == 'Regressor':

                        params = {'n_estimators': [100, 300, 600],
                                  'criterion': ['mse', 'mae'],
                                  'max_depth': [None, 5, 10]}

                    else:

                        params = {'n_estimators': [100, 300, 600],
                                  'criterion': ['gini', 'entropy'],
                                  'max_depth': [None, 5, 10]}

                if Best_clf == 'Gradient Boosting':

                    if self.type_esti == 'Regressor':

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

                    if self.type_esti == 'Regressor':

                        params = {'n_estimators': [10, 100, 300],
                                  'max_depth': [5, 10, None],
                                  'criterion': ['mse', 'mae']}

                    else:

                        params = {'n_estimators': [10, 100, 300],
                                  'max_depth': [5, 10, None],
                                  'criterion': ['gini', 'entropy']}

                elif Best_clf == 'Decision Tree':

                    if self.type_esti == 'Regressor':

                        params = {'max_depth': [5, 10, 50, None],
                                  'criterion': ['mse', 'friedman_mse', 'mae']}

                    else:

                        params = {'max_depth': [5, 10, 50, None],
                                  'criterion': ['gini', 'entropy']}


                elif Best_clf == 'XGBoost':

                    if self.type_esti == 'Classifier':

                        params = {'eta': [.01, .1, .3],
                                  'max_depth': [5, 10, 15],
                                  'gamma': [0, .1, .01]}
                    else:
                        params = {'eta': [.01, .1, .3],
                                  'max_depth': [5, 10, 15],
                                  'gamma': [0, .1, .01]}

                elif Best_clf == 'Bagging':

                    params = {'n_estimators': [100, 300, 600]}

                elif Best_clf == 'KNN':

                    params = {'n_neighbors': [2, 5, 10, 30, 40],
                              'p': [1, 2]}

                elif Best_clf == 'SVM':

                    params = {'C': [1, .5, .1, 5],
                              'tol': [.01, .001, .1, .0001]}

            else:

                if Best_clf == 'Extra Tree':

                    if self.type_esti == 'Regressor':

                        params = {'n_estimators': [10, 100, 300, 600, 1000, 1200],
                                  'criterion': ['mae', 'mse'],
                                  'max_depth': [None, 5, 10, 15, 20, 25]}

                    else:

                        params = {'n_estimators': [10, 100, 300, 600, 1000, 1200],
                                  'criterion': ['gini', 'entropy'],
                                  'max_depth': [None, 5, 10, 15, 20, 25]}

                if Best_clf == 'Gradient Boosting':

                    if self.type_esti == 'Regressor':

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

                    if self.type_esti == 'Regressor':

                        params = {'n_estimators': [10, 100, 300, 600, 1000, 1200],
                                  'max_depth': [5, 10, 15, 20, 25, None],
                                  'criterion': ['mse', 'mae']}

                    else:

                        params = {'n_estimators': [10, 100, 300, 600, 1000, 1200],
                                  'max_depth': [5, 10, 15, 20, 25],
                                  'criterion': ['gini', 'entropy']}

                elif Best_clf == 'Decision Tree':

                    if params == 'Regressor':

                        params = {'max_depth': [5, 10, 50, 100, None],
                                  'criterion': ['mse', 'friedman_mse', 'mae'],
                                  'splitter': ['best', 'random']}

                    else:

                        params = {'max_depth': [5, 10, 50, 100, None],
                                  'criterion': ['gini', 'entropy'],
                                  'splitter': ['best', 'random']}


                elif Best_clf == 'XGBoost':

                    if self.type_esti == 'Classifier':

                        params = {'eta': [0.001, .01, .1, .3, 1],
                                  'max_depth': [5, 10, 15, 20, 25],
                                  'gamma': [0, .1, .01, .001]}
                    else:
                        params = {'tol': [0.001, .01, .1, .3, 1],
                                  'max_depth': [5, 10, 15, 20, 25],
                                  'gamma': [0, .1, .01, .001]}

                elif Best_clf == 'Bagging':

                    params = {'n_estimators': [100, 300, 600, 1000, 1200, 1500]}

                elif Best_clf == 'KNN':

                    params = {'n_neighbors': [2, 5, 10, 30, 40, 70, 100],
                              'p': [1, 2, 3]}

                elif Best_clf == 'SVM':

                    params = {'C': [1, .5, .1, 5, .01, .001],
                              'tol': [.01, .001, .1, .0001, 1],
                              'kernel': ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']}

            if hard_grid:

                print('\n Searching for the best hyperparametres of {} using hard_grid on {} data among : \n'.format(
                    Best_clf, n_grid))

            else:
                print('\n Searching for the best hyperparametres of {} on {} data among : \n'.format(Best_clf, n_grid))
            print('{} \n'.format(params))

            clf = clfs[max(clfs.keys(), key=(lambda k: clfs[k]['mean']))]['clf']
            self.gr = GridSearchCV(clf, param_grid=params, cv=cv_grid, scoring=scoring,
                                   verbose=1, refit=True, iid=True, n_jobs=-1)

            self.gr.fit(self.X_tr[0:n_grid], np.ravel(self.Y_tr[0:n_grid]))

            print('\n In the end, the best estimator is : {} {}'.format(Best_clf, self.type_esti))

            print('\n Using these hyperparametres : {}'.format(self.gr.best_params_))

            if self.neg_result:
                print('\n With this {} score : {}'.format(loss, - self.gr.best_score_))
            else:
                print('\n With this {} score : {}'.format(loss, self.gr.best_score_))

            self.Decision_Function = self.gr.best_estimator_

            self.best_score = - self.gr.best_score_

            if self.lab_num == None:
                self.lab = self.le.inverse_transform(self.gr.classes_)

        else:
            print('\n Best {} : {}'.format(self.type_esti, Best_clf))



    def Feature_Importances_Tree(self, Train, Target, ID = 'ID', value = 0, n = 1000, figsize = (20, 15), nb_features = 15):
        """
        Display the importance features using Giny Impurity

        :param Train: Train Dataset
        :param Target: Target to predict
        :param ID: name of the ID column
        :param value: value to fill missing values
        :param n: sample size for analysis
        :param figsize: figure size window
        :param nb_features:  display the nb_features most importance features
        """
        Data_transform = self.Transform(Train, value, ID)
        Target_transform = self.Transform(Target, value, ID)

        feat_importances = pd.DataFrame()

        if self.type_esti == 'Classifier' :
            clf = ExtraTreesClassifier(n_estimators = 100)
        else:

            clf = ExtraTreesRegressor()

        clf.fit(Data_transform[0:n], np.ravel(Target_transform[0:n]))

        feat_importances['features'] = Data_transform.columns[0:nb_features]
        feat_importances['Score'] = clf.feature_importances_[0:nb_features]
        feat_importances.sort_values(by=['Score'], ascending=False, inplace=True)

        sns.set(font_scale=2)
        plt.subplots(figsize=figsize)
        sns.barplot(x="Score", y="features", data=feat_importances)


    def Feature_Importances_Test(self, Train, Target, ID='ID', value=0, n=1000, nb_features=15,
                                 test_used = f_classif):
        """
        Display the importance features using Giny Impurity

        :param Train: Train Dataset
        :param Target: Target to predict
        :param ID: name of the ID column
        :param value: value to fill missing values
        :param n: sample size for analysis
        :param figsize: figure size window
        :param nb_features:  display the nb_features most importance features
        :param test_used: test type to use
        """
        Train_Transform = self.Transform(Train, ID=ID, value=value)
        Target_Transform = self.Transform(Target, ID=ID, value=value)

        featureScores = pd.DataFrame()
        featureScores['Features'] = Train_Transform.columns


        Test = SelectKBest(score_func= test_used, k=nb_features)
        Test.fit(Train_Transform[0:n], np.ravel(Target_Transform[0:n]))

        featureScores[test_used.__name__] = Test.scores_

        featureScores.sort_values(by=[test_used.__name__], ascending=False, inplace=True)
        featureScores.reset_index(drop=True, inplace=True)


        return(featureScores[0:nb_features])



    def get_highest_corr_target(self, Data, Target, ID='ID', value=0, n = 500, nb_features = 15):
        """
        Display the most correlated fearures with the Target
        :param Data: Dataset
        :param Target: Target to predict
        :param ID: name of the ID column
        :param value: value to fill missing values
        :param n: sample size for analysi
        :param nb_features:  display the nb_features most importance features
        """

        Data_tr = self.Transform(Data, ID=ID, value=value)
        Target_tr = self.Transform(Target, ID=ID, value=value)

        df = pd.DataFrame()

        corr = []
        feature = Data_tr.columns

        for i in Data_tr.columns:

            corr.append(np.abs(pearsonr(np.ravel(Target_tr[Target_tr.columns][0:n]),np.ravel(Data_tr[i][0:n]))[0]))

        df['features'] = feature
        df['Target Correlation'] = corr

        df.sort_values(by=['Target Correlation'], ascending=False, inplace=True)
        df.drop(0, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return(df[0:nb_features])


    def get_highest_corr(self, Data, ID = 'ID', value = 0, n_pairs = 8, n = 500):
        """
        Display the most correlated features pair
        :param Data : Dataset
        :param ID: name of the ID column
        :param value: value to fill missing values
        :param n: sample size for analysi
        :param nb_pairs:  display the nb_pairs most correlated features pair
        """

        Data_tr = self.Transform(Data, ID = ID, value = value)

        df = pd.DataFrame()

        feature1 = []
        feature2 = []
        corr = []

        for i in Data_tr.columns:
            for j in Data_tr.columns:
                feature1.append(i)
                feature2.append(j)
                corr.append(np.abs(pearsonr(Data_tr[i][0:n], Data_tr[j][0:n])[0]))

        df['feature_1'] = feature1
        df['feature_2'] = feature2
        df['correlation_abs'] = corr

        df.sort_values(by=['correlation_abs'], ascending=False, inplace=True)
        df.drop_duplicates(subset='correlation_abs', inplace=True)
        df.drop(0, inplace=True)
        df.reset_index(drop=True, inplace=True)


        return (df[0:n_pairs])


    def corr_mat(self, Train, Target, ID = 'ID', value = 0, figsize = (20, 15), n = 1000, n_pairs = 8):
        """
        Display the (color) correlated matrix, including the Target
        :param Train: Train Dataset
        :param Target: Target Dataset
        :param ID: ID column name
        :param value: value for filling missing values
        :param figsize: figure size
        :param n: sample size for analysis
        :param n_pairs: number of features pair to display
        """

        if n_pairs != None :

            DF = self.get_highest_corr(Train, ID = ID, value = value, n_pairs = n_pairs)

            Target_transform = self.Transform(Target[0:n], value, ID)

            Feature_1 = np.array(DF['feature_1'])
            Feature_2 = np.array(DF['feature_2'])
            target = Target_transform.columns

            Features = np.unique(np.r_[Feature_1, Feature_2, target])


            Data_transform = self.Transform(Train[0:n], value, ID)


            Data_transform[target] = Target_transform

            corrmat = Data_transform[Features].corr()
            top_corr_features = corrmat.index
            sns.set(font_scale=2)
            plt.figure(figsize=figsize)
            sns.heatmap(Data_transform[top_corr_features].corr(), annot=True, cmap="RdYlGn")

        else :

            Target_transform = self.Transform(Target[0:n], value, ID)
            Data_transform = self.Transform(Train[0:n], value, ID)

            target = Target_transform.columns

            Data_transform[target] = Target_transform[target]

            corrmat = Data_transform.corr()
            top_corr_features = corrmat.index
            plt.figure(figsize=figsize)
            sns.heatmap(Data_transform[top_corr_features].corr(), annot=True, cmap="RdYlGn")



    def Transform(self, Data, value=0, ID='ID'):

        """
        Transform all object features of a dataset into numeric ones,
        drop the ID column (None if non present) and fill the missing values.

        :param Data: the dataset to transform
        :param value: value for filling missing values
        :param ID: name of the ID column
        """

        Test = Data.copy()

        if ID != None:
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
        return (Test)


    def custom_grid(self, Train, Target, ID='ID', target_ID=True,
                    n=1000, metric='roc_auc', params=None, cv=3, DF=None, value=0):
        """
        Perform a GridSearchCV with a custom set of hyperparametres and custom estimator

        :param Train: training dataset
        :param Target: target dataset
        :param ID: ID column of the train
        :param target_ID: if True, drop the target ID column
        :param n: size of the sample
        :param metric: loss function used for evaluate perfomance
        :param params: set of hyperparametres
        :param cv: fold number for cross validation
        :param DF: estimator used, if None use the best one found in fit method
        :param value: value for missing values
        """

        target = Target.copy()

        for i in target.columns:
            if target[i].dtype == object:
                le = LabelEncoder()
                le.fit(list(target[i]))
                target[i] = le.transform(list(target[i]))

        if ID != None:
            train = self.Transform(Train, ID=ID, value=value)
            if target_ID:
                target.drop([ID], axis=1, inplace=True)
        if DF == None:
            DF = self.Decision_Function

        gr = GridSearchCV(DF, param_grid=params, cv=cv, scoring=metric, n_jobs=-1,
                          verbose=1, refit=True, iid=True);

        gr.fit(train[0:n], np.ravel(target[0:n]))

        print('\n Best hyperparametres : {}'.format(gr.best_params_))

        if self.neg_result:
            print('\n Giving this {} score : {}'.format(metric, - gr.best_score_))

        else :
            print('\n Giving this {} score : {}'.format(metric, gr.best_score_))




    def Bagg_fit(self, Train, Target, n_estimators = None, n = 1000,
                 cv = 3, value = 0, ID = None, metric = None):
        """
        Use a Bagging algorithm on the best estimator found in fit method

        :param Train: Training dataset
        :param Target: Target dataset
        :param n_estimators: List of estimators to check
        :param n: number of sample to use
        :param cv: folds number to use in GridSearchCV
        :param value: value for filling missing values
        :param ID: ID column
        :param metric: loss score
        """

        params = {'n_estimators' : n_estimators}

        train = self.Transform(Train, value = value, ID = ID)
        target = self.Transform(Target, value = value, ID = ID)

        Best_DF = self.Decision_Function

        if self.type_esti == 'Classifier':
            esti = BaggingClassifier(base_estimator = Best_DF)
        elif self.type_esti == 'Regressor':
            esti = BaggingRegressor(base_estimator = Best_DF)


        DF =  GridSearchCV(estimator = esti, param_grid = params, n_jobs = -1, verbose = 1, cv = cv, scoring = metric)

        DF.fit(train[0:n], np.ravel(target[0:n]))

        print('\n Best hyperparametres : {}'.format(DF.best_params_))

        if self.neg_result:
            print('\n Giving this {} score : {}'.format(metric, - DF.best_score_))
        else:
            print('\n Giving this {} score : {}'.format(metric, DF.best_score_))

        if self.gr.best_score_ < DF.best_score_:
            self.Decision_Function = DF.best_estimator_

        if self.neg_result:
            if self.gr.best_score_ > DF.best_score_:
                self.Decision_Function = DF.best_estimator_



    def pred_grid(self, Test, ID='ID', value=0):

        """
        Predict a target from a dataset using the best hyperparamatres of the GridsSearchCV found in fit method.

        :param Test: The dataset to predict
        :param ID: ID column of the test dataset
        :param value: value for filling missing values
        """

        pred = pd.DataFrame()

        test = self.Transform(Test, ID=ID, value=value)

        if self.type_esti == 'Classifier':
            if self.lab_num:
                predict = self.gr.predict(test)
            else:
                predict = self.le.inverse_transform(self.gr.predict(test))
        else:
            predict = self.gr.predict(test)

        if ID == None:
            pred['Target'] = predict
        else:
            pred[ID] = Test[ID]
            pred['Target'] = predict

        return (pred)



    def pred(self, Test, ID=None, value=0, n=1000, refit = False):
        """
        Predict a target from a dataset using the best estimator from the fit method without using the GridSearchCV prediction

        :param Test: Dataset to predict
        :param ID: ID column
        :param value: Value for filling missing values
        :param n: sample number for fitting
        :param refit: if True, refit on the n sample else, use the first fit model
        :return:
        """

        test = self.Transform(Test, ID=ID, value=value)
        pred = pd.DataFrame()

        self.estim = self.Decision_Function.fit(self.Data[0:n], np.ravel(self.Target[0:n]))

        if refit :
            self.estim = self.Decision_Function.fit(self.Data[0:n], np.ravel(self.Target[0:n]))


        if self.type_esti == 'Classifier':
            if self.lab_num:
                predict = self.gr.predict(test)
            else:
                predict = self.le.inverse_transform(self.estim.predict(test))
        else:
            predict = self.estim.predict(test)

        if ID == None:
            pred['Target'] = predict
        else:
            pred[ID] = Test[ID]
            pred['Target'] = predict

        return(pred)



    def best_size(self, n, metric = 'accuracy_score'):

        """
        Check the best sample size to check the overfitting issues

        :param n: list of size
        :param metric: loss score
        """

        sc_reg = 0
        sc_cla = 0
        size = 0

        if self.type_esti == 'Regressor':
            estim = self.Decision_Function
            estim.fit(self.X_tr[0:n[0]], np.ravel(self.Y_tr[0:n[0]]))
            pred = estim.predict(self.X_te)
            sc_reg = class_for_name('sklearn.metrics', metric)(np.ravel(self.Y_te), np.ravel(pred))
            n=n[1:]

        for i in n:
            estim = self.Decision_Function
            print('Fitting {} datas...'.format(i))
            estim.fit(self.X_tr[0:i],np.ravel(self.Y_tr[0:i]))
            pred = estim.predict(self.X_te)
            score = class_for_name('sklearn.metrics',metric)(np.ravel(self.Y_te), np.ravel(pred))
            print('{} datas -> {} = {} \n'.format(i, metric, score))

            if self.type_esti  == 'Classifier':
                if sc_cla < score:
                    sc_cla = score
                    size = i
            else:
                if sc_reg > score:
                    sc_reg = score
                    size = i

        if self.type_esti == 'Regressor':
            s = sc_reg
        else:
            s = sc_cla

        print('\n In the end, the best data size is {} \n'.format(size))
        print(' With this {} : {}'.format(metric, s))


    def pred_grid_proba(self, Test, ID = 'ID', value = 0):
        """
        Predict classes probabilities from Test dataset

        :param Test: Dataset to predict
        :param ID: The ID column of the Test dataset
        :param ID_pred: If True, add an ID column to the prediction
        :param value: Value for filling missing values in the Test
        """

        test = self.Transform(Test, ID = ID, value = value)

        if self.lab_num:
            pred = pd.DataFrame(self.gr.predict_proba(test), columns=self.gr.classes_)
            pred.insert(loc=0, column=ID, value=Test[ID])
        else:
            pred = pd.DataFrame(self.gr.predict_proba(test), columns=self.le.inverse_transform(self.gr.classes_))
            pred.insert(loc=0, column=ID, value=Test[ID])

        return(pred)


    def pred_proba(self, Test, ID = 'ID', value = 0, n = 1000, refit = True):

        """
        Predict classes probabilities with the refit best estimator found in the fit method

        :param Test: Dataset to predict
        :param ID_Test: ID column of the Test
        :param ID_pred: if True, add an ID column to the prediction
        :param value: value for filling missing values
        :param n: size of the dataset for the new fit
        :param refit: if True, refit at every method launch
        """

        test = self.Transform(Test, ID = ID, value = value)

        if self.estim == None:
            self.estim = self.Decision_Function.fit(self.Data[0:n], np.ravel(self.Target[0:n]))

        if refit:
            self.estim = self.Decision_Function.fit(self.Data[0:n], np.ravel(self.Target[0:n]))


        if self.lab_num:
            pred = pd.DataFrame(self.estim.predict_proba(test), columns=self.estim.classes_)
            pred.insert(loc=0, column=ID, value=Test[ID])
        else:
            pred = pd.DataFrame(self.estim.predict_proba(test), columns=self.le.inverse_transform(self.estim.classes_))
            pred.insert(loc=0, column=ID, value=Test[ID])

        return(pred)

