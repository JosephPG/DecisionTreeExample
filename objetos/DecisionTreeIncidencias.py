import numpy as np
import pandas as pd
from .ManageFileDot import ManageFileDot
from .ManagePlot import ManagePlot
from sklearn.model_selection import train_test_split


class DecisionTreeIncidencias(ManagePlot, ManageFileDot):

    def __init__(self, file_train, file_name_dot='arbol.dot', max_depth=None,
                 *args, **kwargs):
        super(DecisionTreeIncidencias, self).__init__(file_name=file_name_dot,
                                                      max_depth=max_depth,
                                                      *args, **kwargs)
        self.__file_train = self._root_path + file_train
        self.__cargar_data()

    def __cargar_data(self):
        data_csv = pd.read_csv(self.__file_train)
        feature_data = data_csv[data_csv.columns[:-1]].as_matrix()
        target_data = data_csv[data_csv.columns[-1]].as_matrix()
        self._feature_name = data_csv.columns[:-1].tolist()
        self._target_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        self._x_train = feature_data
        self._x_test = feature_data
        self._y_train = target_data
        self._y_test = target_data
        self._entrenar_arbol()
