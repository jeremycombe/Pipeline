class FeatureEngineering(object):

    def __init__(self,
                 Data):

        self.Data = Data
        self.Dict = None


    def Unique(self, view = True):

        self.Dict = {}
        DF_unique = self.Data.copy()
        #DF_unique.fillna(0, inplace = True)

        for i in DF_unique.columns:

            if DF_unique[i].dtype == object or DF_unique[i].dtype == 'bool':
                self.Dict[i] = DF_unique[i].unique()
                if view:
                    print('\n {} : \n \n {} \n'.format(i, np.array(self.Dict[i])))



    def To_numeric(self, freq=True, columns='all'):

        if self.Dict is None:
            self.Unique(view=False)

        DF = self.Data.copy()
        n = DF.shape[0]


        if columns == 'all':
            if freq:

                for i in self.Dict.keys():
                    print(i)

                    T = [np.nan] * n

                    for j in range(len(self.Dict[i])):

                        for k in range(n):
                            if DF[i].iloc[k] == self.Dict[i][j]:
                                T[k] = DF[i][DF[i] == self.Dict[i][j]].count() / n
                    DF[i] = T

        else :

            for i in columns:

                for j in self.Dict.keys():

                    if j == i :

                        print('True')