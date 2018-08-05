import numpy
from .ManageFileDot import ManageFileDot
from .ManagePlot import ManagePlot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class DecisionTreeFirst(ManageFileDot, ManagePlot):

    def __init__(self, file_name_dot='arbol.dot', *args, **kwargs):
        super(DecisionTreeFirst, self).__init__(file_name=file_name_dot, *args, **kwargs)
        self.__iris = load_iris()
        self.__cargar_data()

    def __cargar_data(self):
        """Obtiene nombre del target y caracteristicas,
        separa y obtiene la data de entrenamiento y testeo,
        inicia entrenamiento"""
        self._feature_name = self.__iris.feature_names
        self._target_name = self.__iris.target_names
        self._x_train, \
        self._x_test, \
        self._y_train, \
        self._y_test = \
            train_test_split(self.__iris.data, self.__iris.target)
        self._entrenar_arbol()

    def abrir_dot(self):
        self._crear_dot()
        self._leer_dot()

    def graficar_caracteristicas_importantes(self):
        self._caracteristicas_importante(self._x_train.shape[1])

