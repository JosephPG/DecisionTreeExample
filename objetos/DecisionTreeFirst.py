import numpy
from .ManageFileDot import ManageFileDot
from .ManagePlot import ManagePlot
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class DecisionTreeFirst(ManageFileDot, ManagePlot):

    def __init__(self, file_name_dot='arbol.dot'):
        ManageFileDot.__init__(self, file_name_dot)
        self.arbol = DecisionTreeClassifier()
        self._iris = load_iris()
        self._x_train = numpy.empty(shape=[0,0])
        self._x_test = numpy.empty(shape=[0,0])
        self._y_train = numpy.empty(shape=[0,0])
        self._y_test = numpy.empty(shape=[0,0])
        self.__cargar_data()

    def __cargar_data(self):
        """Separa y obtiene la data de entrenamiento y testeo"""
        self._x_train, \
        self._x_test, \
        self._y_train, \
        self._y_test = \
            train_test_split(self._iris.data, self._iris.target)

    def entrenar_arbol(self):
        """Entrana al arbol"""
        self.arbol.fit(self._x_train, self._y_train)

    def imprimir_resultados(self):
        """Imprime el resultado, de score de acierto y prediccion"""
        train_score_n = self.arbol.score(self._x_train, self._y_train)
        test_score_n = self.arbol.score(self._x_test, self._y_test)
        predict_result = self.arbol.predict(self._x_test)
        print('======================')
        print('Train Score: {}'.format(train_score_n))
        print('Test Score: {}'.format(test_score_n))
        print('Prediccion: {}'.format(predict_result))
        print('======================')

    def generar_dot(self):
        """Crea archivo dot"""
        self._crear_dot(self.arbol, self._iris.target_names,
                        self._iris.feature_names)

    def barra_feature_important(self):
        self._caracteristicas_importante(self._x_train, self._iris.feature_names, self.arbol)

