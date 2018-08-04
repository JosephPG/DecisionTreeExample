import numpy as np
import pandas as pd
from sklearn.datasets  import load_breast_cancer, load_iris

class DecisionTreeExampleZero:
    """Prueba de como funciona el tree
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    ['setosa', 'versicolor', 'virginica'] = [0, 1, 2]"""

    def __init__(self):
        self.iris = load_iris()
        self.to_panda = self.__iris_to_panda()

    def __iris_to_panda(self):
        feature_names = self.iris.feature_names
        feature_names = self.__delete_space(self.__delete_character(feature_names))
        return pd.DataFrame(data=np.c_[self.iris.data, self.iris.target],
                            columns=[feature_names + ['target']])

    def __delete_space(self, array_names):
        return [x.replace(' ', '_') for x in array_names]

    def __delete_character(self, array_names):
        return [x.replace(' (cm)', '') for x in array_names]

    def feature_for_setosa(self):
        """La caracteristica unica de Setosa es el tamaño del
        petalo que es < a 2 cm"""
        for x in self.to_panda.petal_length.values:
            if x < 2:
                print('Iris es Setosa')
            else:
                print('Iris es virginica o versicolor')

    def print_data_col(self):
        """La ultima columna es del resultado, lo cual se obtiene
        solo de las caracteristicas, despues se recorre para obtener
        cada valor de cada columna"""
        for x in self.to_panda.columns[:3]:
            for y in self.to_panda[x]:
                print(y)
