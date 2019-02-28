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
