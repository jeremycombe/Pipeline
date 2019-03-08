import pandas as pd
import numpy as np
import operator
import seaborn as sns
import matplotlib.pyplot as plt
#from preprocessing import scale
#import matplotlib.pyplot.figure as fig
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




class FeatureEngineering(object):

    def __init__(self,
                 Data):

        self.Data = Data
        self.Data_feat = Data.copy()
        self.Dict = None
        self.Dict_count = None


    def Missing_Values(self, Data_base = True):

        print("Missing Values :\n")

        if Data_base :
            Data = self.Data
        else:
            Data = self.Data_feat

        total = Data.isnull().sum().sort_values(ascending=False)
        percent = (Data.isnull().sum() / Data.isnull().count()).sort_values(ascending=False) * 100
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
        print("{} \n".format(missing_data[(percent > 0)]))

    def Unique(self, view = True, Data_base = True):

        self.Dict = {}
        #DF_unique = self.Data.copy()
        #DF_unique.fillna(0, inplace = True)

        if Data_base:

            for i in self.Data.columns:

                if self.Data[i].dtype == object or self.Data[i].dtype == 'bool':
                    self.Dict[i] = self.Data[i].unique()
                    if view:
                        print('\n {} : \n \n {} \n'.format(i, np.array(self.Dict[i])))

        else:

            Empty = True

            for i in self.Data_feat.columns:

                if self.Data_feat[i].dtype == object or self.Data_feat[i].dtype == 'bool':
                    self.Dict[i] = self.Data_feat[i].unique()
                    Empty = False
                    if view:
                        print('\n {} : \n \n {} \n'.format(i, np.array(self.Dict[i])))

            if Empty:

                print('All columns are numerics')




    def To_numeric_freq(self, columns='all'):

        if self.Dict is None:
            self.Unique(view=False)

        #DF = self.Data_feat
        n = self.Data_feat.shape[0]


        if columns == 'all':

            for i in self.Dict.keys():

                T = [np.nan] * n

                for j in range(len(self.Dict[i])):

                    for k in range(n):
                        if self.Data_feat[i].iloc[k] == self.Dict[i][j]:
                            T[k] = self.Data_feat[i][self.Data_feat[i] == self.Dict[i][j]].count() / n
                self.Data_feat[i] = T

        else :

            for i in columns:

                for j in self.Dict.keys():

                    if j == i :

                       T = [np.nan] * n

                       for k in range(len(self.Dict[j])):

                           for l in range(n):

                               if self.Data_feat[i].iloc[l] == self.Dict[j][k]:
                                   T[l] = self.Data_feat[i][self.Data_feat[i] == self.Dict[j][k]].count() / n
                       self.Data_feat[i] = T

        return(self.Data_feat)



    def To_numeric_quant(self, columns = 'all'):

        if self.Dict is None:
            self.Unique(view = False)


        if columns == 'all':

            iteration = self.Dict.keys()

        else:

            iteration = columns

        for i in iteration:
            T = []
            A = pd.DataFrame(self.Data[i].value_counts()).reset_index()
            n = A.shape[0]

            for j in range(n):
                T.append(n - j)

            for l in range(n):

                self.Data_feat[i].replace(A['index'][l], T[l], inplace = True)

        return(self.Data_feat)



    def OneHotEncoder(self, columns):

        self.Data_feat = pd.get_dummies(self.Data_feat, columns = columns)

        return(self.Data_feat)


    def To_numeric_custom(self, Dict_custom):

        """
        :param Dict_custom = {"column" : ['name_column'], "categ" : ["categ1", "categ2", "etc..."],
                                        "to_numeric" :  [1,2,3]}


        Dict_custom = {"column" : ['PRICECLUB_STATUS'], "categ" :['UNSUBSCRIBED', 'REGULAR', 'GOLD', 'PLATINUM', 'SILVER', 0] ,
                                        "to_numeric" :  [0,1,2,3,4,5]}
        """

        n = self.Data_feat.shape[0]
        T = [np.nan] * n

        column_to_transform = np.array(self.Data_feat[Dict_custom['column']])
        categ = Dict_custom['categ']
        to_numeric = Dict_custom['to_numeric']


        for i in range(n):

            for j in range(len(categ)):

                if column_to_transform[i] == categ[j]:

                    T[i] = to_numeric[j]

        self.Data_feat[Dict_custom['column']] = T

        return(self.Data_feat)



    def Plot(self, feature1, feature2, Data_base = True, figsize = (20, 15), n = 1000):

        if Data_base :
            data = self.Data[0:n]
        else:
            data = self.Data_feat[0:n]
        sns.set(font_scale=2)
        plt.figure(figsize=figsize)
        sns.lineplot(x = feature1, y = feature2, markers=True, dashes=False,data = data)
