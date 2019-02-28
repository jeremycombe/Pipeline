class FeatureEngineering(object):

    def __init__(self,
                 Data):

        self.Data = Data
        self.Data_feat = Data.copy()
        self.Dict = None


    def Unique(self, view = True, Data_base = True):

        self.Dict = {}
        DF_unique = self.Data.copy()
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


    # def Dict_dist(self):
    #
    #
    #     if self.Dict is None:
    #         self.Unique(view=False)
    #
    #     Dict_count = {} #self.Dict
    #
    #     #DF = self.Data_feat
    #     n = self.Data_feat.shape[0]
    #
    #     for i in self.Dict.keys():
    #         #print(i)
    #         for j in self.Dict[i]:
    #
    #             T = []
    #             T.append(self.Data_feat[i][self.Data_feat[i] == j].count())
    #
    #         Dict_count[i] = T
    #
    #
    #     return(Dict_count)
    #
    #
    #
    #
    #
    # def To_numeric_quant(self, column = 'all'):
    #
    #
    #     if self.Dict is None:
    #         self.Unique(view=False)
    #
    #     #DF = self.Data_feat
    #     n = self.Data_feat.shape[0]
    #
    #
    #     if columns == 'all':
    #
    #         for i in self.Dict.keys():
    #
    #             T = [np.nan] * n
    #
    #             for j in range(len(self.Dict[i])):
    #
    #                 for k in range(n):
    #                     if self.Data_feat[i].iloc[k] == self.Dict[i][j]:
    #                         T[k] = self.Data_feat[i][self.Data_feat[i] == self.Dict[i][j]].count() / n
    #             self.Data_feat[i] = T
    #
    #     else :
    #
    #         for i in columns:
    #
    #             for j in self.Dict.keys():
    #
    #                 if j == i :
    #
    #                    T = [np.nan] * n
    #
    #                    for k in range(len(self.Dict[j])):
    #
    #                        for l in range(n):
    #
    #                            if self.Data_feat[i].iloc[l] == self.Dict[j][k]:
    #                                T[l] = self.Data_feat[i][self.Data_feat[i] == self.Dict[j][k]].count() / n
    #                    self.Data_feat[i] = T
    #
    #     return(self.Data_feat)





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



